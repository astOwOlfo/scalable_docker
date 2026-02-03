import asyncio

from scalable_docker.client import (
    create_in_clustetr_docker_registry,
    ScalableDockerClient,
    Image,
    create_kubernetes_cluster_with_civo,
    install_docker,
    install_kubectl,
    Container,
)


async def main() -> None:
    # await install_docker()
    # await install_kubectl()
    # await create_kubernetes_cluster_with_civo(n_nodes=4)
    # await create_in_clustetr_docker_registry()
    images = [Image("FROM ubuntu:latest"), Image("FROM alpine:latest")]
    client = ScalableDockerClient()
    # await client.build_images(images)
    containers: list[Container] = await client.start_containers(
        [image.dockerfile_content for image in images]
    )


if __name__ == "__main__":
    asyncio.run(main())
