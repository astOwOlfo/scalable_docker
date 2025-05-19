FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/pallets/werkzeug.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.11
RUN uv pip install pytest pytest-timeout pytest-xprocess cryptography watchdog ephemeral-port-reserve 'MarkupSafe>=2.1.1'
RUN git reset --hard 8ca0b2e3dc2101b95585518d85448d08b3b63568 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
