FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/pydantic/pydantic.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install pytest==5.4.3 pytest-cov==2.10.0 pytest-mock==3.2.0 pytest-sugar==0.9.4 typing-extensions==3.7.4.1 email-validator==1.1.1 python-dotenv==0.14.0
RUN git reset --hard aaec3c9c15581d2de31bcda0fbf096638386ee20 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
