FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/encode/starlette.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install requests pytest aiofiles graphene itsdangerous jinja2 python-multipart pyyaml ujson
RUN git reset --hard 4c621d58c0ce2f111fee4745f944ea6417f1cb53 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
