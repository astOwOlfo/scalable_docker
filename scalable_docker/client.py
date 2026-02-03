from hashlib import sha256
from uuid import uuid4
from time import perf_counter
import json
import base64
import subprocess
from tqdm.asyncio import tqdm as tqdm_asyncio
from os import makedirs
import os
from shlex import quote
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal
from beartype import beartype


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


TIMED_OUT_PROCESS_OUTPUT = ProcessOutput(exit_code=124, stdout="", stderr="timed out")


@beartype
async def run_command(*command: str, assert_success: bool = True) -> ProcessOutput:
    print("Running", command)
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


@beartype
async def install_kubectl() -> None:
    await run_command(
        "bash",
        "-c",
        """curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl""",
    )
    print("INSTALLED DOCKER. PLEASE RESTART THE TERMINAL. EXITING")
    exit()


@beartype
async def install_docker() -> None:
    await run_command(
        "bash",
        "-c",
        'sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce && sudo usermod -aG docker $USER && sudo usermod -aG docker $USER',
    )


@beartype
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
    # TODO: finish installing


@beartype
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


@beartype
async def delete_kubernetes_cluster_with_civo(
    cluster_name: str = "my-k8s-cluster",
) -> None:
    await run_command("civo", "kubernetes", "delete", cluster_name, "-y")


@beartype
async def create_in_clustetr_docker_registry() -> None:
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


@beartype
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


@beartype
async def push_image(dockerfile_content: str) -> None:
    await run_command(
        "docker", "push", f"ghcr.io/astowolfo/{image_name(dockerfile_content)}:latest"
    )


@beartype
async def image_already_pushed(dockerfile_content: str) -> bool:
    output = await run_command(
        "docker",
        "manifest",
        "inspect",
        image_name(dockerfile_content),
        assert_success=False,
    )
    return output.exit_code == 0


@beartype
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
        "tail",
        "-f",
        "/dev/null",
    )

    await run_command(
        "kubectl",
        "patch",
        "deployment",
        deployment_name,
        "-p",
        '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-secret"}]}}}}',
    )


@beartype
async def delete_kubernetes_deployment(deployment_name: str) -> None:
    await run_command("kubectl", "delete", "deployment", deployment_name)


@beartype
async def get_all_kubernetes_deployment_names() -> list[str]:
    output = await run_command("kubectl", "get", "deployments", "-o", "json")
    json_output = json.loads(output.stdout)
    names = [deployment["metadata"]["name"] for deployment in json_output["items"]]
    assert all(isinstance(name, str) for name in names)
    return names


@beartype
async def delete_all_scalable_docker_kubernetes_deployments() -> None:
    deployment_names: list[str] = await get_all_kubernetes_deployment_names()
    deployment_names = [
        name for name in deployment_names if name.startswith("deployment-")
    ]
    await asyncio.gather(
        *[delete_kubernetes_deployment(name) for name in deployment_names]
    )


@beartype
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


@beartype
def random_deployment_name() -> str:
    return f"deployment-{uuid4()}"


@beartype
@dataclass(slots=True)
class ScalableDockerClient:
    containers: list[Container] = field(init=False)
    deployment_names: list[str] = field(init=False)
    deployment_is_ready: list[bool] = field(init=False)
    lock: asyncio.Lock = asyncio.Lock()
    stage: Literal["stopped", "starting", "running", "stopping"] = "stopped"
    stop_task: asyncio.Task = field(init=False)

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

        already_pushed: list[bool] = await asyncio.gather(
            *[
                image_already_pushed(image_name(dockerfile_content))
                for dockerfile_content in dockerfile_contents
            ]
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

    async def start_containers(self, dockerfile_contents: list[str]) -> list[Container]:
        if self.stage == "running":
            await self.start_destroying_containers()

        if self.stage == "stopping":
            await self.stop_task
            async with self.lock:
                self.stage = "stopped"

        async with self.lock:
            assert self.stage == "stopped", (
                "you cannot call start_containers twice in parallel"
            )
            self.stage = "starting"

        self.deployment_names = [random_deployment_name() for _ in dockerfile_contents]

        await asyncio.gather(
            *[
                create_kubernetes_deployment(
                    deployment_name=deployment_name,
                    dockerfile_content=dockerfile_content,
                )
                for deployment_name, dockerfile_content in zip(
                    self.deployment_names, dockerfile_contents, strict=True
                )
            ]
        )

        self.containers = [
            Container(dockerfile_content=dockerfile_content, index=i, worker_index=-1)
            for i, dockerfile_content in enumerate(dockerfile_contents)
        ]
        self.deployment_is_ready = [False] * len(dockerfile_contents)

        async with self.lock:
            self.stage = "running"

        await asyncio.sleep(10.0)

        return self.containers

    async def start_destroying_containers(self) -> None:
        assert self.stage in ["starting", "running"], (
            "you must call start_containers before calling start_destroying_containers"
        )

        async def workload() -> None:
            await asyncio.gather(
                *[delete_kubernetes_deployment(name) for name in self.deployment_names]
            )

        async with self.lock:
            self.stage = "stopping"
            self.stop_task = asyncio.create_task(workload())

    async def run_single_command(
        self, command: str, container: Container, timeout_seconds: float
    ) -> ProcessOutput:
        assert self.stage == "running", (
            "you must call start_containers before calling run_single_command"
        )

        longer_timeout = 2 * timeout_seconds + 8
        return await run_command(
            "timeout",
            str(longer_timeout),
            "kubectl",
            "exec",
            f"deployment/{self.deployment_names[container.index]}",
            "--",
            "timeout",
            str(timeout_seconds),
            "/bin/bash",
            "-c",
            command,
            assert_success=True,
        )

    async def run_commands(
        self,
        container: Container,
        commands: list[str],
        timeout: MultiCommandTimeout,
        blocking: bool = False,
    ) -> list[ProcessOutput]:
        assert self.stage == "running", (
            "you must call start_containers before calling run_single_command"
        )

        if blocking:
            raise NotImplementedError("blocking=True is not supported")

        deployment_name: str = self.deployment_names[container.index]

        if not self.deployment_is_ready[container.index]:
            await wait_for_deployment_ready(deployment_name)

        async with self.lock:
            self.deployment_is_ready[container.index] = True

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
                    timeout_seconds=min(remaining_time, timeout.seconds_per_command),
                )
            )

        return outputs


@beartype
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
