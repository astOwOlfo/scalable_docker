import base64
from shlex import quote
import os
from dataclasses import dataclass, asdict
from typing import Any
from beartype import beartype

from scalable_docker.rest_client_base import AsyncJsonRESTClient


@beartype
@dataclass(frozen=True, slots=True)
class ProcessOutput:
    exit_code: int
    stdout: str
    stderr: str


@beartype
@dataclass(frozen=True, slots=True)
class Image:
    dockerfile_content: str
    max_cpus: float | int = 1
    max_memory_gigabytes: float | int = 1.0


@beartype
@dataclass(frozen=True, slots=True)
class Container:
    dockerfile_content: str
    index: int
    worker_index: int


@beartype
@dataclass(frozen=True, slots=True)
class ScalableDockerServerError(Exception):
    server_response: Any


@beartype
@dataclass(frozen=True, slots=True)
class MultiCommandTimeout:
    seconds_per_command: int | float
    total_seconds: int | float


@beartype
class ScalableDockerClient(AsyncJsonRESTClient):
    key: str

    def __init__(
        self, *args, key: str, server_url: str | None = None, **kwargs
    ) -> None:
        if server_url is None:
            server_url = os.environ.get("SCALABLE_DOCKER_SERVER_URL")

        if server_url is None:
            raise ValueError(
                "A sever url must be provided to RemoteDockerSandbox, either by passing a `server_url` argument to its constructor or by setting the `SCALABLE_DOCKER_SERVER_URL` system variable."
            )

        super().__init__(*args, server_url=server_url, **kwargs)

        self.key = key

    def is_error(self, server_response: Any) -> bool:
        return isinstance(server_response, dict) and "error" in server_response.keys()

    async def docker_prune_everything(self) -> Any:
        print("Pruning docker images...")

        response = await self.call_server(function="docker_prune_everything")

        if self.is_error(response):
            raise ScalableDockerServerError(response)

        print("Done pruning!")

    async def build_images(
        self,
        images: list[Image],
        batch_size: int | None = None,
        max_attempts: int = 1,
        workers_per_dockerfile: int | None = None,
        ignore_errors: bool = False,
    ) -> None:
        print(f"Building {len(images)} images...")

        response = await self.call_server(
            function="build_images",
            key=self.key,
            images=[asdict(image) for image in images],
            batch_size=batch_size,
            max_attempts=max_attempts,
            workers_per_dockerfile=workers_per_dockerfile,
            ignore_errors=ignore_errors,
        )

        if self.is_error(response):
            raise ScalableDockerServerError(response)

        print("Done building images!")

    async def number_healthy_workers(self) -> int:
        response = await self.call_server(function="number_healthy_workers")

        if self.is_error(response):
            raise ScalableDockerServerError(response)

        return response

    async def start_containers(self, dockerfile_contents: list[str]) -> list[Container]:
        response = await self.call_server(
            function="start_containers",
            key=self.key,
            dockerfile_contents=dockerfile_contents,
        )

        if self.is_error(response):
            raise ScalableDockerServerError(response)

        return [Container(**container) for container in response]

    async def start_destroying_containers(self) -> None:
        response = await self.call_server(
            key=self.key, function="start_destroying_containers"
        )

        if self.is_error(response):
            raise ScalableDockerServerError(response)

    async def run_commands(
        self,
        container: Container,
        commands: list[str],
        timeout: MultiCommandTimeout,
    ) -> list[ProcessOutput]:
        response = await self.call_server(
            function="run_commands",
            key=self.key,
            container=asdict(container),
            commands=commands,
            total_timeout_seconds=timeout.seconds_per_command,
            per_command_timeout_seconds=timeout.total_seconds,
        )

        if self.is_error(response):
            raise ScalableDockerServerError(response)

        return [ProcessOutput(**output) for output in response]


@beartype
def upload_file_command(filename: str, content: str) -> str:
    encoded_data = base64.b64encode(content.encode()).decode()
    return f"echo {encoded_data} | base64 -d > {quote(filename)}"
