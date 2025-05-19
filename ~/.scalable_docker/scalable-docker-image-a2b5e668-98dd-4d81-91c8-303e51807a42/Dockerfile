FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python3-venv python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /testbed/
RUN python -m venv .venv
RUN echo '. .venv/bin/activate' >> ~/.bashrc
RUN .venv/bin/python -m pip install pytest hypothesis parsy
RUN mkdir assignment
