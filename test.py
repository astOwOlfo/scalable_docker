import json
import asyncio

from scalable_docker.client import (
    MultiCommandTimeout,
    create_in_clustetr_docker_registry,
    ScalableDockerClient,
    Image,
    create_kubernetes_cluster_with_civo,
    delete_all_scalable_docker_kubernetes_deployments,
    delete_kubernetes_cluster_with_civo,
    install_docker,
    install_kubectl,
    Container,
)


async def main() -> None:
    # await install_docker()
    # await install_kubectl()
    # await delete_kubernetes_cluster_with_civo()
    # await create_kubernetes_cluster_with_civo(n_nodes=4)
    # await create_in_clustetr_docker_registry()
    await delete_all_scalable_docker_kubernetes_deployments()

    images = [
        Image(json.loads(line)["dockerfile_content"])
        for line in open("final-hard.jsonl")
        if line.strip() and not line.startswith("#")
    ]
    images = images[:64]
    client = ScalableDockerClient(key="")
    await client.build_images(images, batch_size=64)
    containers: list[Container] = await client.start_containers(
        [image.dockerfile_content for image in images]
    )
    for container in containers:
        print(
            "OUTPUT:",
            await client.run_commands(container, ["pwd"], MultiCommandTimeout(8, 8)),
        )
    await client.start_destroying_containers()


if __name__ == "__main__":
    asyncio.run(main())
