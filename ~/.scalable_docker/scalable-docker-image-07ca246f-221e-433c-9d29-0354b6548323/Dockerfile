FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/aws/aws-cli.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install botocore==1.2.10 colorama==0.3.3 'docutils>=0.10' rsa==3.3.0 mock==1.0.1 'six>=1.10.0' 'pytest>=6.0.0'
RUN git reset --hard 449c0c05c7c28c5e9a4e29aae40fa6c47e0e43ea && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
