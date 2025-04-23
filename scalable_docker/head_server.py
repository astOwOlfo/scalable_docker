from argparse import ArgumentParser
import random
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any
from beartype import beartype

from scalable_docker.rest_server_base import JsonRESTServer
from scalable_docker.worker_client import WorkerClient, ResourceUsage


MAX_HEALTH = 1.0


@beartype
@dataclass(slots=True)
class Worker:
    client: WorkerClient
    health: float = MAX_HEALTH
    estimated_resource_usage: ResourceUsage = field(
        default_factory=lambda: ResourceUsage(
            n_running_containers=0,
            n_existing_containers=0,
            fraction_memory_used=0.1,
            fraction_disk_used=0.1,
        )
    )

    def health_and_resource_availability_score(
        self, running_container_capacity: int, existing_container_capacity: int
    ) -> float | int:
        return self.health * self.estimated_resource_usage.resource_availability_score(
            running_container_capacity=running_container_capacity,
            existing_container_capacity=existing_container_capacity,
        )


@beartype
@dataclass(frozen=True, slots=True)
class Exponential:
    initial_value: float | int
    base: float | int

    def __call__(self, x: float | int) -> float | int:
        return self.initial_value * self.base**x


@beartype
@dataclass
class HeadServer(JsonRESTServer):
    host: str
    port: int
    worker_urls: list[str]
    health_decay: float = 1e-3
    max_retries_creating_sandbox_without_waiting: int = 8
    max_retries_creating_sandbox_with_waiting: int = 8
    wait_before_retry_creating_sandbox_seconds: Exponential = Exponential(
        initial_value=2.0, base=2.0
    )
    running_container_capacity_per_worker: int = 32
    existing_container_capacity_per_worker: int = 64
    probability_worker_load_check: float = 1e-1
    workers: list[Worker] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(port=self.port, host=self.host)

        self.workers = [
            Worker(client=WorkerClient(server_url=url)) for url in self.worker_urls
        ]

    def functions_exposed_through_api(self) -> dict[str, Callable]:
        return {
            "create_sandbox": self.create_sandbox,
            "run_commands": self.run_commands,
            "cleanup_sandbox": self.cleanup_sandbox,
            "add_worker": self.add_worker,
        }

    def choose_worker(self) -> Worker:
        assert len(self.workers) > 0

        scores = [
            worker.health_and_resource_availability_score(
                running_container_capacity=self.existing_container_capacity_per_worker,
                existing_container_capacity=self.existing_container_capacity_per_worker,
            )
            for worker in self.workers
        ]
        total_score = sum(scores)

        if total_score == 0:
            probabilities_of_choosing_each_worker = [1.0] * len(self.workers)
            print(
                "WARNING: ALL WORKERS HAVE HEALTH AND RESOURCE AVAILABILITY SCORE 0. THIS IS MOST PROBABLY INDICATIVE THAT ALL WORKER SERVERS ARE DOWN OR OVERLOADED TO THE POINT WHERE THEY STOPPED WORKING"
            )
        else:
            probabilities_of_choosing_each_worker = [
                score / total_score for score in scores
            ]

        worker = random.choices(
            self.workers, weights=probabilities_of_choosing_each_worker
        )[0]

        self.decay_worker_healths()

        if random.uniform(0, 1) <= self.probability_worker_load_check:
            resource_usage = worker.client.get_resource_usage()
            if resource_usage is not None:
                worker.estimated_resource_usage = resource_usage

        return worker

    def decay_worker_healths(self) -> None:
        for worker in self.workers:
            worker.health = (
                1 - self.health_decay
            ) * worker.health + self.health_decay * MAX_HEALTH

    def create_sandbox(
        self,
        container_name: str,
        dockerfile_content: str,
        max_memory_gb: float | int | None,
        max_cpus: int | None,
        max_lifespan_seconds: int | None,
    ) -> Any:
        total_retries = (
            self.max_retries_creating_sandbox_with_waiting
            + self.max_retries_creating_sandbox_without_waiting
        )

        if len(self.workers) == 0:
            return {
                "error": "You must add at least one worker url before being able to create a sandbox."
            }

        for i_retry in range(total_retries):
            worker = self.choose_worker()

            response = worker.client.create_sandbox(
                container_name=container_name,
                dockerfile_content=dockerfile_content,
                max_memory_gb=max_memory_gb,
                max_cpus=max_cpus,
                max_lifespan_seconds=max_lifespan_seconds,
            )

            print(f"Create sandbox response: {response}")

            if not self.server_response_is_failure(response):
                worker.estimated_resource_usage.n_running_containers += 1
                return response

            worker.health = 0.0

            if i_retry >= self.max_retries_creating_sandbox_without_waiting:
                time.sleep(
                    self.wait_before_retry_creating_sandbox_seconds(
                        i_retry - self.max_retries_creating_sandbox_without_waiting
                    )
                )
                print(
                    f"WARNING: Failed to find worker that could successfully create a sandbox after {i_retry} retries."
                )

        print(
            f"WARNING: Failed to find a worker server that could successfully create a sandbox after {total_retries} retries. Gave up creating the sandbox."
        )

        return {"error": f"Failed to create sandbox after {total_retries} retries."}

    def run_commands(
        self,
        container_name: str,
        commands: list[str],
        total_timeout_seconds: float | int,
        per_command_timeout_seconds: float | int,
    ) -> Any:
        print("HeadServer.run_commands called")

        worker = self.choose_worker()

        response = worker.client.run_commands(
            container_name=container_name,
            commands=commands,
            total_timeout_seconds=total_timeout_seconds,
            per_command_timeout_seconds=per_command_timeout_seconds,
        )

        print(f"{response=}")

        if self.server_response_is_failure(response):
            worker.health = 0.0

        return response

    def cleanup_sandbox(self, container_name: str) -> Any:
        worker = self.choose_worker()

        response = worker.client.cleanup_sandbox(container_name=container_name)

        if self.server_response_is_failure(response):
            worker.health = 0.0
        else:
            worker.estimated_resource_usage.n_running_containers -= 1

        return response
    
    def add_worker(self, worker_server_url: str) -> Any:
        worker = Worker(client=WorkerClient(server_url=worker_server_url))
        response = worker.client.get_resource_usage()
        
        if self.server_response_is_failure(response):
            return response
    
        self.worker_urls.append(worker_server_url)
        self.workers.append(worker)

    def server_response_is_failure(self, response: Any) -> bool:
        return isinstance(response, dict) and "error" in response.keys()


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
        "--health-decay", type=float, default=1e-3, help="TO DO: Write help message."
    )
    parser.add_argument(
        "--max-retries-creating-sandbox-without-waiting",
        type=int,
        default=8,
        help="When failed to create a sandbox on a worker server, retry with a different server.",
    )
    parser.add_argument(
        "--max-retries-creating-sandbox-with-waiting",
        type=int,
        default=8,
        help="After --max-retries-creating-sandbox-without-waiting retries, do more retries with an exponentially increasing delay.",
    )
    parser.add_argument(
        "--wait-before-retry-creating-sandbox-seconds-initial-value",
        type=float,
        default=2.0,
        help="How long to wait during the first one of the --max-retries-creating-sandbox-without-waiting retries.",
    )
    parser.add_argument(
        "--wait-before-retry-creating-sandbox-seconds-base",
        type=float,
        default=2.0,
        help="At every subsequent one of the --max-retries-creating-sandbox-without-waiting retries, multiply the waiting time by this much.",
    )
    parser.add_argument(
        "--running-container-capacity-per-worker",
        default=32,
        help="If the estimated number of containers running on a worker server goes above `running_container_capacity_per_worker / 2`, it will be less chosen when creating a new sandbox, with the probability decreasing with the number of containers running on it.",
    )
    parser.add_argument(
        "--existing-container-capacity-per-worker",
        default=64,
        help="Same as --running-container-capacity-per-worker but also include containers that are not running but have not been deleted (i.e. the containers shown in the output of `docker ps -a`, not only those shown in the output of `docker ps`).",
    )
    parser.add_argument(
        "--probability-worker-load-check",
        type=float,
        default=1e-1,
        help="Each time a sandbox is created, do an extra API call to the worker used for it to get how many resources (e.g. ram, disk) it has with this probability. Workers with fewer resources will be chosen less often.",
    )
    args = parser.parse_args()

    server = HeadServer(
        host=args.host,
        port=args.port,
        worker_urls=args.worker_urls,
        health_decay=args.health_decay,
        max_retries_creating_sandbox_without_waiting=args.max_retries_creating_sandbox_without_waiting,
        max_retries_creating_sandbox_with_waiting=args.max_retries_creating_sandbox_with_waiting,
        wait_before_retry_creating_sandbox_seconds=Exponential(
            initial_value=args.wait_before_retry_creating_sandbox_seconds_initial_value,
            base=args.wait_before_retry_creating_sandbox_seconds_base,
        ),
        running_container_capacity_per_worker=args.running_container_capacity_per_worker,
        existing_container_capacity_per_worker=args.existing_container_capacity_per_worker,
        probability_worker_load_check=args.probability_worker_load_check,
    )

    server.serve()


if __name__ == "__main__":
    main_cli()
