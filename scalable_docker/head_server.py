from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from collections.abc import Callable
from typing import Any, TypedDict
from beartype import beartype

from scalable_docker.worker_server import Image
from scalable_docker.rest_server_base import JsonRESTServer
from scalable_docker.rest_client_base import JsonRESTClient


@beartype
class Container(TypedDict):
    dockerfile_content: str
    index: int
    worker_index: int


@beartype
class Worker:
    url: str
    client: JsonRESTClient
    last_error_time: float | None

    def __init__(self, url: str) -> None:
        self.url = url
        self.client = JsonRESTClient(url)
        self.last_error_time = None


# TO DO: LOCKS!!


@beartype
class HeadServer(JsonRESTServer):
    workers: list[Worker]
    running_containers: list[Container] | None = None
    delay_before_retrying_worker_after_error_seconds: float | int

    def __init__(
        self,
        worker_urls: list[str],
        host: str = "0.0.0.0",
        port: int = 8080,
        delay_before_retrying_worker_after_error_seconds: float | int = 3600,
    ) -> None:
        super().__init__(host=host, port=port)
        self.workers = [Worker(url) for url in worker_urls]
        self.running_containers = None
        self.delay_before_retrying_worker_after_error_seconds = (
            delay_before_retrying_worker_after_error_seconds
        )

    def functions_exposed_through_api(self) -> dict[str, Callable]:
        return {
            "build_images": self.build_images,
            "push_built_images_to_docker_hub": self.push_built_images_to_docker_hub,
            "number_healthy_workers": self.number_healthy_workers,
            "start_containers": self.start_containers,
            "start_destroying_containers": self.start_destroying_containers,
            "run_commands": self.run_commands,
        }

    def is_error(self, worker_server_response: Any) -> bool:
        return (
            isinstance(worker_server_response, dict)
            and "error" in worker_server_response.keys()
        )

    def build_images(
        self,
        images: list[Image],
        prune: bool,
        batch_size: int | None,
        max_attempts: int,
        pull_from_docker_hub: bool,
        docker_hub_username: str | None,
    ) -> Any:
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    worker.client.call_server,
                    function="build_images",
                    images=images,
                    prune=prune,
                    batch_size=batch_size,
                    max_attempts=max_attempts,
                    pull_from_docker_hub=pull_from_docker_hub,
                    docker_hub_username=docker_hub_username,
                )
                for worker in self.workers
            ]
            responses = [future.result() for future in futures]

        unsuccessful_responses: list[dict] = []
        for worker, response in zip(self.workers, responses, strict=True):
            if self.is_error(response):
                worker.last_error_time = perf_counter()
                unsuccessful_responses.append(response)
            else:
                worker.last_error_time = None

        if len(unsuccessful_responses) > 0:
            return {
                "error": f"{len(unsuccessful_responses)} out of {len(self.workers)} failed when trying to build images. All the responses from the workers which failed are: {unsuccessful_responses}"
            }

    def push_built_images_to_docker_hub(self, docker_hub_username: str, docker_hub_access_token: str) -> Any:
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    worker.client.call_server,
                    function="push_built_images_to_docker_hub",
                    docker_hub_username=docker_hub_username,
                    docker_hub_access_token=docker_hub_access_token,
                )
                for worker in self.workers
            ]
            responses = [future.result() for future in futures]

        unsuccessful_responses: list[dict] = []
        for worker, response in zip(self.workers, responses, strict=True):
            if self.is_error(response):
                worker.last_error_time = perf_counter()
                unsuccessful_responses.append(response)
            else:
                worker.last_error_time = None

        if len(unsuccessful_responses) > 0:
            return {
                "error": f"{len(unsuccessful_responses)} out of {len(self.workers)} failed when trying to build images. All the responses from the workers which failed are: {unsuccessful_responses}"
            }

    def healthy_worker_indices(self) -> list[int]:
        return [
            i
            for i, worker in enumerate(self.workers)
            if worker.last_error_time is None
            or worker.last_error_time
            <= perf_counter() + self.delay_before_retrying_worker_after_error_seconds
        ]

    def number_healthy_workers(self) -> int:
        return len(self.healthy_worker_indices())

    def start_containers(
        self, dockerfile_contents: list[str]
    ) -> list[Container] | dict:
        if self.running_containers is not None:
            self.start_destroying_containers()

        healthy_worker_indices = self.healthy_worker_indices()

        if len(healthy_worker_indices) == 0:
            return {"error": "There are no healthy workers."}

        container_indices_by_worker: list[list[int]] = [
            [] for _ in range(len(healthy_worker_indices))
        ]
        dockerfile_contents_by_worker: list[list[str]] = [
            [] for _ in range(len(healthy_worker_indices))
        ]
        for i, dockerfile_content in enumerate(dockerfile_contents):
            container_indices_by_worker[i % len(healthy_worker_indices)].append(i)
            dockerfile_contents_by_worker[i % len(healthy_worker_indices)].append(
                dockerfile_content
            )

        with ThreadPoolExecutor(max_workers=len(healthy_worker_indices)) as executor:
            futures = [
                executor.submit(
                    self.workers[i_healthy_worker].client.call_server,
                    function="start_containers",
                    dockerfile_contents=dockerfile_contents_for_worker,
                )
                for i_healthy_worker, dockerfile_contents_for_worker in zip(
                    healthy_worker_indices, dockerfile_contents_by_worker, strict=True
                )
            ]

            responses = [future.result() for future in futures]

        unsuccessful_responses: list[dict] = []
        containers: list[Container | None] = [None] * len(dockerfile_contents)
        for i_healthy_worker, response, container_indices in zip(
            healthy_worker_indices, responses, container_indices_by_worker, strict=True
        ):
            if self.is_error(response):
                self.workers[i_healthy_worker].last_error_time = perf_counter()
                unsuccessful_responses.append(response)
                continue
            for i, container in enumerate(response):
                assert containers[container_indices[i]] is None
                containers[container_indices[i]] = container | {
                    "worker_index": i_healthy_worker
                }

        if len(unsuccessful_responses) > 0:
            return {
                "error": f"{len(unsuccessful_responses)} out of {len(healthy_worker_indices)} workers failed when trying to start containers. The failed responses are: {unsuccessful_responses}"
            }

        assert all(container is not None for container in containers)

        self.running_containers = containers  # type: ignore

        return containers  # type: ignore

    def start_destroying_containers(self) -> Any:
        if self.running_containers is None:
            return

        used_worker_indices = set(
            container["worker_index"] for container in self.running_containers
        )

        self.running_containers = None

        with ThreadPoolExecutor(max_workers=len(used_worker_indices)) as executor:
            futures = [
                executor.submit(
                    self.workers[i].client.call_server,
                    function="start_destroying_containers",
                )
                for i in used_worker_indices
            ]

            responses = [future.result() for future in futures]

        unsuccessful_responses = [
            response for response in responses if self.is_error(response)
        ]

        if len(unsuccessful_responses) > 0:
            return {
                "error": f"{len(unsuccessful_responses)} out of {len(used_worker_indices)} workers failed when trying to stop containers. All the failed responses are: {unsuccessful_responses}"
            }

    def run_commands(
        self,
        container: Container,
        commands: list[str],
        total_timeout_seconds: float | int,
        per_command_timeout_seconds: float | int,
    ) -> list[dict]:
        worker = self.workers[container["worker_index"]]

        response = worker.client.call_server(
            function="run_commands",
            container={
                "dockerfile_content": container["dockerfile_content"],
                "index": container["index"],
            },
            commands=commands,
            total_timeout_seconds=total_timeout_seconds,
            per_command_timeout_seconds=per_command_timeout_seconds,
        )

        if self.is_error(response):
            worker.last_error_time = perf_counter()

        return response


@beartype
def main_cli() -> None:
    parser = ArgumentParser(
        description="Start server that will archestrate worker servers running docker sandboxes. Its url should then be given to `scalable_docker.client.RemoteDockerSandbox`. The url is 'https://this_machines_ip:port/'."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--worker-urls",
        type=lambda urls: urls.split(","),
        default=[],
        help="Comma separated list of worker server urls.",
    )
    parser.add_argument(
        "--delay-before-retrying-worker-after-error-seconds", type=int, default=3600
    )
    args = parser.parse_args()

    server = HeadServer(
        host=args.host,
        port=args.port,
        worker_urls=args.worker_urls,
        delay_before_retrying_worker_after_error_seconds=args.delay_before_retrying_worker_after_error_seconds,
    )

    server.serve()


if __name__ == "__main__":
    main_cli()
