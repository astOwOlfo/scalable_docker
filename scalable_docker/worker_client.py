from dataclasses import dataclass
from typing import Any
from beartype import beartype

from scalable_docker.rest_client_base import JsonRESTClient


@beartype
@dataclass(slots=True)
class ResourceUsage:
    n_running_containers: int
    n_existing_containers: int
    fraction_memory_used: float | int
    fraction_disk_used: float | int

    def resource_availability_score(
        self, running_container_capacity: int, existing_container_capacity: int
    ) -> float | int:
        score = (1 - self.fraction_memory_used) * (1 - self.fraction_disk_used)
        if self.n_running_containers > running_container_capacity / 2:
            score *= running_container_capacity / 2 / self.n_running_containers
        if self.n_existing_containers > existing_container_capacity / 2:
            score *= existing_container_capacity / 2 / self.n_existing_containers
        return score


@beartype
class WorkerClient(JsonRESTClient):
    def server_response_is_error(self, response: Any) -> bool:
        return isinstance(response, dict) and "error" in response.keys()

    def create_sandbox(
        self,
        container_name: str,
        dockerfile_content: str,
        max_memory_gb: float | int | None,
        max_cpus: int | None,
        max_lifespan_seconds: int | None,
    ) -> Any:
        return self.call_server(
            function="create_sandbox",
            container_name=container_name,
            dockerfile_content=dockerfile_content,
            max_memory_gb=max_memory_gb,
            max_cpus=max_cpus,
            max_lifespan_seconds=max_lifespan_seconds,
        )

    def run_commands(
        self,
        container_name: str,
        commands: list[str],
        total_timeout_seconds: float | int,
        per_command_timeout_seconds: float | int,
    ) -> Any:
        return self.call_server(
            function="run_commands",
            container_name=container_name,
            commands=commands,
            total_timeout_seconds=total_timeout_seconds,
            per_command_timeout_seconds=per_command_timeout_seconds,
        )

    def cleanup_sandbox(self, container_name: str) -> Any:
        return self.call_server(
            function="cleanup_sandbox", container_name=container_name
        )

    def get_resource_usage(self) -> ResourceUsage | None:
        response = self.call_server(function="get_resource_usage")

        print(f"WorkerClient.get_resource_usage: {response=}")

        if self.server_response_is_error(response):
            return None

        return ResourceUsage(**response)
