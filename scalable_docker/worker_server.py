from hashlib import sha256
from argparse import ArgumentParser
from pathlib import Path
from os import makedirs, path
from shlex import quote
from shutil import rmtree
import yaml
from tqdm import tqdm
from time import perf_counter
from subprocess import run, PIPE, TimeoutExpired, Popen
from more_itertools import chunked
from dataclasses import asdict
from collections.abc import Callable, Iterable
from typing import TypedDict
from beartype import beartype

from scalable_docker.rest_server_base import JsonRESTServer
from scalable_docker.client import ProcessOutput


@beartype
class Image(TypedDict):
    dockerfile_content: str
    max_cpus: float | int
    max_memory_gigabytes: float | int


@beartype
class Container(TypedDict):
    dockerfile_content: str
    index: int


@beartype
class WorkerServer(JsonRESTServer):
    working_directory: str
    docker_compose_yaml_path: str
    built_dockerfile_contents: list[str] | None
    running_containers: list[Container] | None
    destroy_sandboxes_process: Popen | None

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        working_directory: str = path.join(Path.home(), ".scalable_docker"),
    ) -> None:
        super().__init__(host=host, port=port)
        self.working_directory = working_directory
        self.docker_compose_yaml_path = path.join(
            working_directory, "docker-compose.yaml"
        )
        self.built_dockerfile_contents = None
        self.running_containers = None
        self.destroy_sandboxes_process = None

    def functions_exposed_through_api(self) -> dict[str, Callable]:
        return {
            "build_images": self.build_images,
            "push_built_images_to_docker_hub": self.push_built_images_to_docker_hub,
            "start_containers": self.start_containers,
            "start_destroying_containers": self.start_destroying_containers,
            "run_commands": self.run_commands,
        }

    def build_images(
        self,
        images: list[Image],
        prune: bool,
        batch_size: int | None,
        max_attempts: int,
        pull_from_docker_hub: bool,
        docker_hub_username: str | None,
    ) -> None:
        assert (docker_hub_username is not None) == pull_from_docker_hub

        if self.running_containers is not None:
            self.start_destroying_containers()
            self.wait_until_done_destroying_containers()

        if prune:
            run_and_raise_if_fails(
                ["docker", "system", "prune", "-a", "--volumes", "--force"]
            )

        rmtree(self.working_directory, ignore_errors=True)

        docker_compose_yaml: dict[str, dict] = {"services": {}}
        for image in unique(images):
            image_name = self.image_name(dockerfile_content=image["dockerfile_content"])
            image_directory = path.join(self.working_directory, image_name)
            rmtree(image_directory, ignore_errors=True)
            makedirs(image_directory)
            with open(path.join(image_directory, "Dockerfile"), "w") as f:
                if pull_from_docker_hub:
                    tagged_image_name = self.tagged_image_name(
                        dockerfile_content=image["dockerfile_content"],
                        docker_hub_username=docker_hub_username,  # type: ignore
                    )
                    f.write(f"FROM {tagged_image_name}")
                else:
                    f.write(image["dockerfile_content"])
            docker_compose_yaml["services"][image_name] = {
                "build": image_directory,
                "command": "tail -f /dev/null",
                "deploy": {
                    "resources": {
                        "limits": {
                            "cpus": image["max_cpus"],
                            "memory": f"{image['max_memory_gigabytes']}gb",
                        }
                    }
                },
            }

        with open(self.docker_compose_yaml_path, "w") as f:
            yaml.dump(docker_compose_yaml, f)

        image_names: list[str] = list(docker_compose_yaml["services"].keys())
        batched_image_names: list[list[str]] = (
            list(chunked(image_names, batch_size))
            if batch_size is not None
            else [image_names]
        )
        for i, image_name_batch in enumerate(batched_image_names):
            print(
                f"BUILDING {len(image_name_batch)} IMAGES (BATCH {i + 1} OF {len(batched_image_names)})", flush=True
            )
            for i_attempt in range(max_attempts):
                print(f"ATTMEPT {i_attempt} OUT OF {max_attempts}", flush=True)
                try:
                    run_and_raise_if_fails(
                        [
                            "docker",
                            "compose",
                            "-f",
                            self.docker_compose_yaml_path,
                            "build",
                        ]
                        + image_name_batch
                    )
                    break
                except Exception as e:
                    print(
                        "DOCKER COMPOSE BUILD FAILED WITH THE FOLLOWING EXCEPTION:", e, flush=True
                    )
                    if i_attempt == max_attempts - 1:
                        raise e

        self.built_dockerfile_contents = list(
            set(image["dockerfile_content"] for image in images)
        )

    def push_built_images_to_docker_hub(
        self,
        docker_hub_username: str,
        # docker_hub_access_token: str,
    ) -> None:
        # run_and_raise_if_fails(
        #     f"echo {quote(docker_hub_access_token)} | docker login --username {quote(docker_hub_username)} --password-stdin",
        #     shell=True,
        # )

        for dockerfile_content in tqdm(
            self.built_dockerfile_contents, desc="pushing to docker hub"
        ):
            image_name = "scalable_docker-" + self.image_name(
                dockerfile_content=dockerfile_content
            )
            tagged_image_name = self.tagged_image_name(
                dockerfile_content=dockerfile_content,
                docker_hub_username=docker_hub_username,
            )
            run_and_raise_if_fails(["docker", "tag", image_name, tagged_image_name])
            run_and_raise_if_fails(["docker", "push", tagged_image_name])

    def image_name(self, dockerfile_content: str) -> str:
        return sha256(dockerfile_content.encode()).hexdigest()

    def tagged_image_name(
        self, dockerfile_content: str, docker_hub_username: str
    ) -> str:
        return f"{docker_hub_username}/scalable_docker-{self.image_name(dockerfile_content=dockerfile_content)}"

    def start_containers(self, dockerfile_contents: list[str]) -> list[Container]:
        assert self.built_dockerfile_contents is not None, (
            "You must call build_images before calling start_containers."
        )

        if self.running_containers is not None:
            self.start_destroying_containers()

        self.wait_until_done_destroying_containers()

        assert all(
            dockerfile_content in self.built_dockerfile_contents
            for dockerfile_content in dockerfile_contents
        ), (
            "The dockerfile_contents argument to start_containers should only contain dockerfiles that have been given to the previous call to build_images."
        )

        docker_compose_up_command: list[str] = [
            "docker",
            "compose",
            "-f",
            self.docker_compose_yaml_path,
            "up",
            "-d",
        ]
        for dockerfile_content in self.built_dockerfile_contents:
            image_name = self.image_name(dockerfile_content=dockerfile_content)
            count = dockerfile_contents.count(dockerfile_content)
            docker_compose_up_command += ["--scale", f"{image_name}={count}"]

        run_and_raise_if_fails(docker_compose_up_command)

        containers: list[Container] = []
        for i, dockerfile_content in enumerate(dockerfile_contents):
            image_name = self.image_name(dockerfile_content=dockerfile_content)
            containers.append(
                {
                    "dockerfile_content": dockerfile_content,
                    "index": dockerfile_contents[:i].count(dockerfile_content),
                }
            )

        self.running_containers = containers

        return containers

    def start_destroying_containers(self) -> None:
        assert self.running_containers is not None, (
            "You must call start_containers before each call to destroy_containers."
        )

        self.running_containers = None

        self.destroy_sandboxes_process = Popen(
            [
                "docker",
                "compose",
                "-f",
                self.docker_compose_yaml_path,
                "down",
                "--volumes",
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            errors="replace",
        )

    def wait_until_done_destroying_containers(self) -> None:
        if self.destroy_sandboxes_process is None:
            return

        stdout, stderr = self.destroy_sandboxes_process.communicate()

        print("DOCKER COMPOSE DOWN FINISHED RUNNING", flush=True)
        print("DOCKER COMPOSE DOWN STDOUT:", stdout, flush=True)
        print("DOCKER COMPOSE DOWN STDERR:", stderr, flush=True)

        assert self.destroy_sandboxes_process.returncode == 0, (
            f"Error trying to destroy sandboxes: docker compose down returned with a nonzero exit code.\nExit code: {self.destroy_sandboxes_process.returncode}\nStdout:{stdout}\nStderr:{stderr}"
        )

        self.destroy_sandboxes_process = None

    def run_single_command(
        self, container: Container, command: str, timeout_seconds: float | int
    ) -> ProcessOutput:
        assert (
            self.destroy_sandboxes_process is None
            and self.running_containers is not None
        ), "You must call start_containers before calling run_single_command."

        assert self.destroy_sandboxes_process is None

        assert self.running_containers is not None, (
            "You must call start_containers before calling run_commands."
        )

        assert container in self.running_containers, (
            "The container argument to run_commands must be one of the values returned by the previous call to start_containers."
        )

        try:
            output = run(
                [
                    "docker",
                    "compose",
                    "-f",
                    self.docker_compose_yaml_path,
                    "exec",
                    f"--index={container['index']}",
                    self.image_name(dockerfile_content=container["dockerfile_content"]),
                    "/bin/bash",
                    "-c",
                    command,
                ],
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
        container: Container,
        commands: list[str],
        total_timeout_seconds: float | int,
        per_command_timeout_seconds: float | int,
    ) -> list[dict]:
        assert (
            self.destroy_sandboxes_process is None
            and self.running_containers is not None
        ), "You must call start_containers before calling run_commands."

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

            output = self.run_single_command(
                container=container,
                command=command,
                timeout_seconds=min(per_command_timeout_seconds, remaining_time),
            )

            outputs.append(output)

        return [asdict(output) for output in outputs]


@beartype
def run_and_raise_if_fails(command: list[str] | str, shell: bool = False) -> None:
    assert isinstance(command, str) == shell

    output = run(command, capture_output=True, shell=shell)

    if output.returncode != 0:
        raise ValueError(
            f"Command {command} failed with non-zero exit code {output.returncode}.\n\nSTDOUT: {output.stdout}\n\nSTDERR:{output.stderr}"
        )


@beartype
def unique(xs: Iterable) -> list:
    unique_xs = []
    for x in xs:
        if x not in unique_xs:
            unique_xs.append(x)
    return unique_xs


@beartype
def main_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--working-directory",
        type=str,
        default=path.join(Path.home(), ".scalable_docker"),
    )
    args = parser.parse_args()

    server = WorkerServer(
        host=args.host, port=args.port, working_directory=args.working_directory
    )

    server.serve()


if __name__ == "__main__":
    main_cli()
