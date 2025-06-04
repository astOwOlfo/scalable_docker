from argparse import ArgumentParser
from uuid import uuid4
from os import makedirs, path
from time import perf_counter
from subprocess import run, PIPE, TimeoutExpired
from threading import Thread
from dataclasses import dataclass, asdict
from collections.abc import Callable
from typing import Any
from beartype import beartype

from scalable_docker.rest_server_base import JsonRESTServer
from scalable_docker.client import ProcessOutput


@beartype
@dataclass(frozen=True, slots=True)
class Failure:
    message: str


@beartype
@dataclass(frozen=True, slots=True)
class Dockerfile:
    content: str


@beartype
@dataclass(frozen=True, slots=True)
class Image:
    name: str
    dockerfile: Dockerfile


@beartype
@dataclass(frozen=True, slots=True)
class Container:
    name: str
    image: Image
    startup_commands: list[str]
    max_memory_gb: float | int | None
    max_cpus: float | int | None
    max_lifespan_seconds: float | int | None

    def __hash__(self) -> int:
        return hash(self.name)


@beartype
class WorkerServer(JsonRESTServer):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        working_directory: str = "~/.scalable_docker/",
    ) -> None:
        super().__init__(host=host, port=port)
        self.used_container_names: set[str] = set()
        self.images: dict[Dockerfile, Image] = {}
        self.build_image_threads: dict[Image, Thread] = {}
        self.build_image_failures: dict[Image, Failure] = {}
        self.start_container_threads: dict[Container, Thread] = {}
        self.start_container_failures: dict[Container, Failure] = {}
        self.running_containers_by_name: dict[str, Container] = {}
        self.working_directory = working_directory
        makedirs(working_directory, exist_ok=True)

    def functions_exposed_through_api(self) -> dict[str, Callable]:
        return {
            "create_sandbox": self.create_sandbox,
            "run_commands": self.run_commands,
            "cleanup_sandbox": self.cleanup_sandbox,
            "get_resource_usage": self.get_resource_usage,
        }

    def create_sandbox(
        self,
        container_name: str,
        dockerfile_content: str,
        startup_commands: list[str],
        max_memory_gb: float | int | None,
        max_cpus: float | int | None,
        max_lifespan_seconds: float | int | None,
    ) -> Any:
        if container_name in self.used_container_names:
            return {"error": f"Sandbox id '{container_name}' is already used."}
        self.used_container_names.add(container_name)

        dockerfile = Dockerfile(content=dockerfile_content)
        image: Image | None = self.images.get(dockerfile)
        already_built = image is not None
        if not already_built:
            image = Image(
                name=f"scalable-docker-image-{uuid4()}", dockerfile=dockerfile
            )
            self.images[dockerfile] = image
            build_thread = Thread(target=self.build_image, args=(image,))
            build_thread.start()
            self.build_image_threads[image] = build_thread

        container = Container(
            name=container_name,
            image=image,
            startup_commands=startup_commands,
            max_memory_gb=max_memory_gb,
            max_cpus=max_cpus,
            max_lifespan_seconds=max_lifespan_seconds,
        )
        start_thread = Thread(target=self.start_container, args=(container,))
        self.start_container_threads[container] = start_thread
        start_thread.start()

        self.running_containers_by_name[container.name] = container

    def build_image(self, image: Image) -> None:
        docker_directory = path.join(self.working_directory, image.name)
        makedirs(docker_directory)

        with open(path.join(docker_directory, "Dockerfile"), "w") as f:
            f.write(image.dockerfile.content)

        output = run(["docker", "build", "-t", image.name, docker_directory])

        built_successfully = output.returncode == 0
        if not built_successfully:
            self.build_image_failures[image] = Failure(
                f"Failed building image.\nDocker build exit code: {output.returncode}\n\nDocker build stdout: {output.stdout}\n\nDocker build stderr: {output.stderr}"
            )

    def start_container(self, container: Container) -> None:
        build_thread = self.build_image_threads.get(container.image)
        if build_thread is not None:
            build_thread.join()

        if container.image in self.build_image_failures:
            return

        command = ["docker", "run", "-d", "--rm", "--name", container.name]
        if container.max_memory_gb is not None:
            command += ["--memory", f"{container.max_memory_gb}gb"]
        if container.max_cpus is not None:
            command += ["--cpus", str(container.max_cpus)]
        command += [
            "--tty",
            container.image.name,
            "/bin/bash",
            "-c",
            f"sleep {container.max_lifespan_seconds if container.max_lifespan_seconds is not None else 'infinity'}",
        ]

        output = run(command, stdout=PIPE, stderr=PIPE, text=True, errors="replace")

        started_successfully = output.returncode == 0
        if not started_successfully:
            self.start_container_failures[container] = Failure(
                f"Failed starting container '{container.name}'.\nDocker run exit code: {output.returncode}\n\nDocker run stdout: {output.stdout}\n\nDocker run stderr: {output.stderr}"
            )

        for command in container.startup_commands:
            output = run(["docker", "exec", container.name, "/bin/bash", "-c", command])
            success = output.returncode == 0
            if not success:
                self.start_container_failures[container] = Failure(
                    f"Failed starting container '{container.name}'.\nA startup command exited with nonzero exit code.\n\nFailed command: {command}\n\nExit code:{output.returncode}\n\nStdout{output.stdout}\n\nStderr:{output.stderr}"
                )

    def wait_for_container_to_start(self, container: Container) -> None:
        if container in self.start_container_threads:
            self.start_container_threads[container].join()

    def get_container_or_error(self, container_name: str) -> Container | Failure:
        container = self.running_containers_by_name.get(container_name)
        if container is None:
            return Failure(
                f"The container with name '{container_name}' does not exist, has been stopped, or died."
            )

        self.wait_for_container_to_start(container)

        build_failure = self.build_image_failures.get(container.image)
        if build_failure is not None:
            return build_failure

        start_failure = self.start_container_failures.get(container)
        if start_failure is not None:
            return start_failure

        return container

    def run_single_command(
        self, container: Container, command: str, timeout_seconds: float | int
    ) -> ProcessOutput:
        try:
            output = run(
                ["docker", "exec", container.name, "/bin/bash", "-c", command],
                timeout=timeout_seconds,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                errors="replace",
            )
        except TimeoutExpired:
            return ProcessOutput(exit_code=1, stdout="", stderr="timed out")
        return ProcessOutput(
            exit_code=output.returncode, stdout=output.stdout, stderr=output.stderr
        )

    def run_commands(
        self,
        container_name: str,
        commands: list[str],
        total_timeout_seconds: float | int,
        per_command_timeout_seconds: float | int,
    ) -> Any:
        container = self.get_container_or_error(container_name)
        if isinstance(container, Failure):
            return {"error": container.message}

        start_time = perf_counter()

        outputs: list[ProcessOutput] = []

        for command in commands:
            time_spent_so_far = perf_counter() - start_time
            remaining_time = total_timeout_seconds - time_spent_so_far
            if remaining_time <= 0:
                outputs.append(
                    ProcessOutput(exit_code=1, stdout="", stderr="timed out")
                )
                continue

            with open("/home/user/temp/times-spent.log", "a") as f:
                f.write(f"{container_name=} {time_spent_so_far=} {command[:256]=}\n")

            output = self.run_single_command(
                container=container,
                command=command,
                timeout_seconds=min(per_command_timeout_seconds, remaining_time),
            )

            outputs.append(output)

        return [asdict(output) for output in outputs]

    def cleanup_sandbox(self, container_name: str) -> Any:
        container = self.get_container_or_error(container_name)
        if isinstance(container, Failure):
            return {"error": container.message}

        Thread(target=self.cleanup_sandbox_blocking, args=(container,)).start()

    def cleanup_sandbox_blocking(self, container: Container) -> None:
        run(["docker", "stop", container.name], stdout=PIPE, stderr=PIPE)
        run(["docker", "rm", container.name], stdout=PIPE, stderr=PIPE)

    def get_resource_usage(self) -> Any:
        return {
            "n_running_containers": int(
                run(
                    "docker ps -q | wc -l",
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                    errors="replace",
                    text=True,
                ).stdout.strip()
            ),
            "n_existing_containers": int(
                run(
                    "docker ps -q | wc -l",
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                    errors="replace",
                    text=True,
                ).stdout.strip()
            ),
            "fraction_memory_used": float(
                run(
                    "free | awk '/Mem:/ {printf \"%.2f%%\\n\", $3/$2 * 100}'",
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                    errors="replace",
                    text=True,
                )
                .stdout.strip()
                .removesuffix("%")
            )
            / 100,
            "fraction_disk_used": float(
                run(
                    "df / | awk 'NR==2 {print $5}'",
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                    errors="replace",
                    text=True,
                )
                .stdout.strip()
                .removesuffix("%")
            )
            / 100,
        }


@beartype
def main_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--working-directory", type=str, default="~/.scalable_docker/")
    args = parser.parse_args()

    server = WorkerServer(
        host=args.host, port=args.port, working_directory=args.working_directory
    )

    server.serve()


if __name__ == "__main__":
    main_cli()
