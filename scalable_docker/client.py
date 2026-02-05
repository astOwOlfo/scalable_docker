from contextlib import nullcontext
from hashlib import sha256
from uuid import uuid4
from time import perf_counter
import json
from sys import stderr
import base64
import subprocess
from tqdm.asyncio import tqdm as tqdm_asyncio
from os import makedirs
import os
from shlex import quote
import asyncio
from dataclasses import dataclass, field
from typing import Any


# subprocess.run(["ulimit", "-n", "65536"], check=True)


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class Image:
    dockerfile_content: str
    max_cpus: float | int = 1
    max_memory_gigabytes: float | int = 1.0


ContainerId = int


@dataclass(frozen=True, slots=True)
class Container:
    id: ContainerId
    deployment_name: str
    dockerfile_content: str


@dataclass(frozen=True, slots=True)
class ScalableDockerServerError(Exception):
    server_response: Any


@dataclass(frozen=True, slots=True)
class MultiCommandTimeout:
    seconds_per_command: int | float
    total_seconds: int | float


TIMED_OUT_PROCESS_OUTPUT = ProcessOutput(exit_code=124, stdout="", stderr="timed out")


async def run_command(*command: str, assert_success: bool = True) -> ProcessOutput:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    assert process.returncode is not None
    if assert_success and process.returncode != 0:
        raise RuntimeError(
            f"Command {command} failed. Exit code: {process.returncode}\nSTDOUT: {stdout.decode(errors='replace')}\nSTDERR: {stderr.decode(errors='replace')}"
        )
    return ProcessOutput(
        exit_code=process.returncode,
        stdout=stdout.decode(errors="ignore"),
        stderr=stderr.decode(errors="ignore"),
    )


async def install_kubectl() -> None:
    await run_command(
        "bash",
        "-c",
        """curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl""",
    )


async def install_docker() -> None:
    await run_command(
        "bash",
        "-c",
        'sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce && sudo usermod -aG docker $USER && sudo usermod -aG docker $USER',
    )
    print(
        "INSTALLED DOCKER. PLEASE RESTART THE TERMINAL. IF USING TMUX, RUN `tmux kill-server`. EXITING"
    )
    exit()


async def install_civo() -> None:
    already_installed: bool = (
        await run_command("civo", "--version", assert_success=False)
    ).exit_code == 0
    if already_installed:
        return

    install_command_output = await run_command(
        "bash", "-c", "curl -sL https://civo.com/get | sh"
    )
    print(
        "INSTALL COMMAND OUTPUT:",
        install_command_output.stdout + install_command_output.stderr,
    )
    await run_command("sudo", "mv", "/tmp/civo /usr/local/bin/civo")


async def create_kubernetes_secret(github_token: str) -> None:
    await run_command(
        "kubectl",
        "create",
        "secret",
        "docker-registry",
        "ghcr-secret",
        "--docker-server=ghcr.io",
        "--docker-username=astowolfo",
        f"--docker-password={github_token}",
        "--docker-email=volodimir1024@gmail.com",
    )


async def create_kubernetes_cluster_with_civo(
    n_nodes: int,
    instance_type: str = "g4s.kube.large",
    cluster_name: str = "my-k8s-cluster",
) -> None:
    await run_command(
        "civo",
        "kubernetes",
        "create",
        cluster_name,
        "--nodes",
        str(n_nodes),
        "--size",
        instance_type,
        "--wait",
    )
    await run_command("civo", "kubernetes", "config", cluster_name, "--save")


async def delete_kubernetes_cluster_with_civo(
    cluster_name: str = "my-k8s-cluster",
) -> None:
    await run_command("civo", "kubernetes", "delete", cluster_name, "-y")


async def create_in_cluster_docker_registry() -> None:
    await run_command(
        "kubectl",
        "create",
        "deployment",
        "registry",
        "--image=registry:2",
        "--port=5000",
    )
    # await run_command(
    #     "kubectl", "expose", "deployment", "registry", "--port=5000", "--type=ClusterIP"
    # )
    await run_command(
        "kubectl",
        "expose",
        "deployment",
        "registry",
        "--port=5000",
        "--type=NodePort",
        "--name=registry-np",
    )
    await asyncio.sleep(10.0)
    subprocess.Popen(
        ["kubectl", "port-forward", "svc/registry-np", "5000:5000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def image_name(dockerfile_content: str) -> str:
    hash: str = sha256(dockerfile_content.encode()).hexdigest()
    return f"image-{hash[:32]}"


async def build_image(dockerfile_content: str) -> None:
    dir: str = os.path.abspath(
        os.path.join("dockerfiles", image_name(dockerfile_content))
    )
    makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    await run_command(
        "docker",
        "build",
        "-t",
        f"ghcr.io/astowolfo/{image_name(dockerfile_content)}:latest",
        dir,
    )


async def push_image(dockerfile_content: str) -> None:
    await run_command(
        "docker", "push", f"ghcr.io/astowolfo/{image_name(dockerfile_content)}:latest"
    )


async def image_already_pushed(dockerfile_content: str) -> bool:
    output = await run_command(
        "docker",
        "manifest",
        "inspect",
        f"ghcr.io/astowolfo/{image_name(dockerfile_content)}",
        assert_success=False,
    )
    return output.exit_code == 0


async def create_kubernetes_deployment(
    deployment_name: str, dockerfile_content: str
) -> None:
    await run_command(
        "kubectl",
        "create",
        "deployment",
        deployment_name,
        f"--image=ghcr.io/astowolfo/{image_name(dockerfile_content)}:latest",
        "--",
        "/bin/bash",
        "-c",
        "tail -f /dev/null",
    )

    await run_command(
        "kubectl",
        "patch",
        "deployment",
        deployment_name,
        "-p",
        '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-secret"}]}}}}',
    )


async def delete_kubernetes_deployment(deployment_name: str) -> None:
    await run_command(
        "kubectl",
        "delete",
        "deployment",
        deployment_name,
        "--wait",
        "--timeout",
        "300s",
        assert_success=False,
    )

    await run_command(
        "kubectl",
        "wait",
        "--for=delete",
        "pod",
        "-l",
        f"app={deployment_name}",
        "--timeout=3600s",
    )


async def get_all_kubernetes_deployment_names() -> list[str]:
    output = await run_command("kubectl", "get", "deployments", "-o", "json")
    json_output = json.loads(output.stdout)
    names = [deployment["metadata"]["name"] for deployment in json_output["items"]]
    assert all(isinstance(name, str) for name in names)
    return names


async def delete_all_scalable_docker_kubernetes_deployments() -> None:
    deployment_names: list[str] = await get_all_kubernetes_deployment_names()
    deployment_names = [
        name for name in deployment_names if name.startswith("deployment-")
    ]
    await asyncio.gather(
        *[delete_kubernetes_deployment(name) for name in deployment_names]
    )


async def wait_for_deployment_ready(
    deployment_name: str, timeout_seconds: int = 3600
) -> None:
    await run_command(
        "kubectl",
        "wait",
        "--for=condition=Available",
        f"deployment/{deployment_name}",
        f"--timeout={timeout_seconds}s",
    )


def random_deployment_name() -> str:
    return f"deployment-{uuid4()}"


# !!! ONLY GLOBAL TEMPORARILY !!!
exec_semaphore: asyncio.Semaphore | None = None


@dataclass(slots=True)
class ScalableDockerClient:
    key: str
    max_parallel_commands: int | None = None
    max_command_length: int = 65536
    # exec_semaphore: asyncio.Semaphore | None = field(init=False)
    containers: dict[ContainerId, Container] = field(default_factory=lambda: {})
    stopped_container_ids: set[int] = field(default_factory=lambda: set())
    wait_for_deployment_ready_tasks: dict[ContainerId, asyncio.Task] = field(
        default_factory=lambda: {}
    )
    stop_tasks: dict[ContainerId, asyncio.Task] = field(default_factory=lambda: {})
    lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())
    create_containers_lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())
    container_id_counter: int = 0

    async def new_container_id(self) -> ContainerId:
        async with self.lock:
            id = self.container_id_counter
            self.container_id_counter += 1
            return id

    async def __post_init__(self) -> None:
        # !!! TEMPORARY !!!
        global exec_semaphore
        if exec_semaphore is None:
            exec_semaphore = asyncio.Semaphore(64)

    # !!! ONLY COMMENTED TEMPORARILY !!!
    # def __post_init__(self) -> None:
    #     self.exec_semaphore = (
    #         asyncio.Semaphore(self.max_parallel_commands)
    #         if self.max_parallel_commands is not None
    #         else None
    #     )

    async def docker_prune_everything(self) -> Any:
        raise NotImplementedError()

    async def build_images(
        self,
        images: list[Image],
        batch_size: int | None = None,
        max_retries: int = 1,
    ) -> None:
        if max_retries != 1:
            raise NotImplementedError("max_retries != 1 is not supported")

        dockerfile_contents = set(image.dockerfile_content for image in images)

        already_pushed: list[bool] = await asyncio_gather_max_parallels(
            *[
                image_already_pushed(dockerfile_content)
                for dockerfile_content in dockerfile_contents
            ],
            max_parallels=256,
            progress_bar_description="checking which images already exist on ghcr.io",
        )

        dockerfile_contents = [
            dockerfile_content
            for dockerfile_content, pushed in zip(dockerfile_contents, already_pushed)
            if not pushed
        ]

        if len(dockerfile_contents) == 0:
            return

        async def build_and_push_image(dockerfile_content: str) -> None:
            await build_image(dockerfile_content)
            await push_image(dockerfile_content)

        await asyncio_gather_max_parallels(
            *[
                build_and_push_image(dockerfile_content)
                for dockerfile_content in dockerfile_contents
            ],
            max_parallels=batch_size
            if batch_size is not None
            else len(dockerfile_contents),
            progress_bar_description="building images",
        )

    async def wait_until_all_containers_stopped(self) -> None:
        for container_id, stop_task in self.stop_tasks.items():
            await stop_task
            async with self.lock:
                self.stopped_container_ids.add(container_id)

    async def start_containers(self, dockerfile_contents: list[str]) -> list[Container]:
        async with self.create_containers_lock:
            await self.start_destroying_containers()
            await self.wait_until_all_containers_stopped()
            assert set(self.stopped_container_ids) == set(self.containers.keys())

            container_ids = [await self.new_container_id() for _ in dockerfile_contents]
            async with self.lock:
                self.containers = {
                    id: Container(
                        id=id,
                        deployment_name=random_deployment_name(),
                        dockerfile_content=dockerfile_content,
                    )
                    for id, dockerfile_content in zip(
                        container_ids, dockerfile_contents, strict=True
                    )
                }

                self.stopped_container_ids = set()
                self.stop_tasks = {}

            await asyncio.gather(
                *[
                    create_kubernetes_deployment(
                        deployment_name=container.deployment_name,
                        dockerfile_content=container.dockerfile_content,
                    )
                    for container in self.containers.values()
                ]
            )

            self.wait_for_deployment_ready_tasks = {
                id: asyncio.create_task(
                    wait_for_deployment_ready(container.deployment_name)
                )
                for id, container in self.containers.items()
            }

            return [self.containers[id] for id in container_ids]

    async def start_destroying_container(self, container: Container) -> None:
        async with self.lock:
            assert container.id in self.containers.keys(), (
                "container has already been destroyed"
            )

            already_stopping = container.id in self.stop_tasks.keys()
            if already_stopping:
                return

            self.stop_tasks[container.id] = asyncio.create_task(
                delete_kubernetes_deployment(container.deployment_name)
            )

    async def start_destroying_containers(self) -> None:
        for container in self.containers.values():
            await self.start_destroying_container(container)

    async def run_single_command(
        self, command: str, container: Container, timeout_seconds: float
    ) -> ProcessOutput:
        assert container.id in self.containers.keys(), (
            "container has already been destroyed"
        )

        await self.wait_for_deployment_ready_tasks[container.id]

        if len(command) > self.max_command_length:
            print(
                f"Scalable Docker Warning: Truncating long command of length {len(command)} to length {self.max_command_length}.",
                file=stderr,
            )
            command = command[:command]

        longer_timeout = 2 * timeout_seconds + 8
        return await run_command(
            "timeout",
            str(longer_timeout),
            "kubectl",
            "exec",
            f"deployment/{container.deployment_name}",
            "--",
            "timeout",
            str(timeout_seconds),
            "/bin/bash",
            "-c",
            command,
            assert_success=False,
        )

    async def run_commands(
        self,
        container: Container,
        commands: list[str],
        timeout: MultiCommandTimeout,
        blocking: bool = False,
    ) -> list[ProcessOutput]:
        if blocking:
            raise NotImplementedError("blocking=True is not supported")

        assert exec_semaphore is not None
        async with exec_semaphore:
            outputs: list[ProcessOutput] = []
            start_time = perf_counter()
            for command in commands:
                time_spent = perf_counter() - start_time
                remaining_time = timeout.total_seconds - time_spent
                if remaining_time <= 0:
                    outputs.append(TIMED_OUT_PROCESS_OUTPUT)
                    continue
                outputs.append(
                    await self.run_single_command(
                        command=command,
                        container=container,
                        timeout_seconds=min(
                            remaining_time, timeout.seconds_per_command
                        ),
                    )
                )

            return outputs


def upload_file_command(filename: str, content: str) -> str:
    encoded_data = base64.b64encode(content.encode()).decode()
    return f"echo {encoded_data} | base64 -d > {quote(filename)}"


async def asyncio_gather_max_parallels(
    *xs, max_parallels: int, progress_bar_description: str | None = None
) -> list:
    semaphore = asyncio.Semaphore(max_parallels)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    wrapped = [run_with_semaphore(coro) for coro in xs]

    return await tqdm_asyncio.gather(*wrapped, desc=progress_bar_description)
