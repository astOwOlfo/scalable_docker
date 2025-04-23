from uuid import uuid4
import base64
from shlex import quote
import os
from typing import Any
from beartype import beartype
from dataclasses import dataclass

from scalable_docker.rest_client_base import JsonRESTClient


@beartype
@dataclass(frozen=True, slots=True)
class ProcessOutput:
    exit_code: int
    stdout: str
    stderr: str


@beartype
@dataclass
class RemoteDockerSandbox(JsonRESTClient):
    def __init__(
        self,
        dockerfile_content: str,
        max_memory_gb: float | int | None = 1.0,
        max_cpus: int | None = 1,
        max_lifespan_seconds: int | None = 10_000,
        server_url: str | None = None,
    ) -> None:
        if server_url is None:
            server_url = os.environ.get("DOCKER_SANDBOX_SERVER_URL")

        if server_url is None:
            raise ValueError(
                "A sever url must be provided to RemoteDockerSandbox, either by passing a `server_url` argument to its constructor or by setting the `DOCKER_SANDBOX_SERVER_URL` system variable."
            )

        super().__init__(server_url=server_url)

        self.container_name = f"sandbox-container-{uuid4()}"

        creation_response = self.call_server(
            function="create_sandbox",
            container_name=self.container_name,
            dockerfile_content=dockerfile_content,
            max_memory_gb=max_memory_gb,
            max_cpus=max_cpus,
            max_lifespan_seconds=max_lifespan_seconds,
        )

        if self.server_response_is_error(creation_response):
            raise ValueError(
                f"Failed creating sandbox. The server response is: {creation_response}"
            )

    def server_response_is_error(self, response: Any) -> bool:
        return isinstance(response, dict) and "error" in response.keys()

    def run_commands(
        self,
        commands: list[str],
        total_timeout_seconds: float | int = 5.0,
        per_command_timeout_seconds: float | int = 4.0,
    ) -> list[ProcessOutput]:
        response = self.call_server(
            function="run_commands",
            container_name=self.container_name,
            commands=commands,
            total_timeout_seconds=total_timeout_seconds,
            per_command_timeout_seconds=per_command_timeout_seconds,
        )

        if self.server_response_is_error(response):
            raise ValueError(
                f"Failed running commands in the sandbox. The server response is: {response}"
            )

        print(f"{response=}")

        return [ProcessOutput(**output) for output in response]

    def cleanup(self) -> None:
        response = self.call_server(
            function="cleanup_sandbox", container_name=self.container_name
        )

        if self.server_response_is_error(response):
            raise ValueError(
                f"Failed sandbox cleanup. The server response is: {response}"
            )

    def upload_file(self, filename: str, content: str) -> ProcessOutput:
        command = RemoteDockerSandbox.upload_file_command(
            filename=filename, content=content
        )
        return self.run_commands([command])[0]

    @staticmethod
    def upload_file_command(filename: str, content: str) -> str:
        encoded_data = base64.b64encode(content.encode()).decode()
        return f"echo {encoded_data} | base64 -d > {quote(filename)}"


@beartype
def add_worker(worker_server_url: str, head_server_url: str | None) -> None:
    if head_server_url is None:
        head_server_url = os.environ.get("DOCKER_SANDBOX_SERVER_URL")

    if head_server_url is None:
        raise ValueError(
            "A sever url must be provided to RemoteDockerSandbox, either by passing a `server_url` argument to its constructor or by setting the `DOCKER_SANDBOX_SERVER_URL` system variable."
        )

    response = JsonRESTClient(head_server_url).call_server(
        function="add_worker", worker_server_url=worker_server_url
    )

    if response is not None:
        message = f"FAILURE ADDING NEW WORKER. Server response: {response}"
        print(f"FAILURE ADDING NEW WORKER. Server response: {response}")
        raise ValueError(message)
