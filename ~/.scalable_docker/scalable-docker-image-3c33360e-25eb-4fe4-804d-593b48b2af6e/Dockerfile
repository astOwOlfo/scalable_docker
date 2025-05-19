FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/networkx/networkx.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install 'numpy>=1.19' 'scipy>=1.8' 'matplotlib>=3.4' 'pandas>=1.3' 'sympy>=1.0' 'pytest>=7.0' 'pytest-cov>=3.0' 'codecov>=2.1'
RUN git reset --hard 12c1a00cd116701a763f7c57c230b8739d2ed085 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
