from hashlib import sha256
import json
import base64
import subprocess
from tqdm import tqdm
from os import makedirs
import os
from shlex import quote
import asyncio
from dataclasses import asdict, dataclass
from typing import Any
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


TIMED_OUT_PROCESS_OUTPUT = ProcessOutput(exit_code=1, stdout="", stderr="timed out")


@beartype
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
async def create_kubernetes_cluster(
    n_nodes: int, instance_type: str = "g4s.kube.large"
) -> None:
    await run_command(
        "civo",
        "kubernetes",
        "create my-k8s-cluster",
        "--nodes",
        str(n_nodes),
        "--size",
        instance_type,
        "--wait",
    )


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
    await run_command(
        "kubectl", "expose", "deployment", "registry", "--port=5000", "--type=ClusterIP"
    )
    await asyncio.sleep(10.0)
    subprocess.Popen(
        ["kubectl", "port-forward", "svc/registry", "5000:5000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def image_name(dockerfile_content: str) -> str:
    hash: str = sha256(dockerfile_content.encode()).hexdigest()
    return f"image-{hash}"


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
        f"localhost:5000/{image_name(dockerfile_content)}:latest",
        dir,
    )


@beartype
async def push_image(dockerfile_content) -> None:
    await run_command(
        "docker", "push", f"localhost:5000/{image_name(dockerfile_content)}:latest"
    )


@beartype
@dataclass(frozen=True, slots=True)
class ScalableDockerClient:
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
        for dockerfile_content in tqdm(dockerfile_contents, desc="building images"):
            await build_image(dockerfile_content)
            await push_image(dockerfile_content)

    async def start_containers(
        self, dockerfile_contents: list[str]
    ) -> list[Container]: ...

    async def start_destroying_containers(self) -> None: ...

    async def run_commands(
        self,
        container: Container,
        commands: list[str],
        timeout: MultiCommandTimeout,
        blocking: bool = False,
    ) -> list[ProcessOutput]: ...


@beartype
def upload_file_command(filename: str, content: str) -> str:
    encoded_data = base64.b64encode(content.encode()).decode()
    return f"echo {encoded_data} | base64 -d > {quote(filename)}"
