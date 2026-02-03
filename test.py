import asyncio

from scalable_docker.client import (
    create_in_clustetr_docker_registry,
    ScalableDockerClient,
    Image,
    install_kubectl,
)


async def main() -> None:
    await install_kubectl()
    await create_in_clustetr_docker_registry()
    images = [Image("FROM ubuntu:latest"), Image("FROM alpine:latest")]
    client = ScalableDockerClient()
    await client.build_images(images)


if __name__ == "__main__":
    asyncio.run(main())
