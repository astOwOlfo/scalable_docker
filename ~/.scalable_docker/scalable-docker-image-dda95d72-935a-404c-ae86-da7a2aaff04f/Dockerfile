FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /testbed/
RUN uv init --python 3.11
RUN uv venv --python 3.11
RUN uv add tabulate
RUN echo '. .venv/bin/activate' >> ~/.bashrc
