FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/Pylons/pyramid.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install setuptools 'WebOb>=1.3.1' 'repoze.lru>=0.4' 'zope.interface>=3.8.0' 'zope.deprecation>=3.5.0' 'venusian>=1.0a3' 'translationstring>=0.4' 'PasteDeploy>=1.5.0' 'WebTest>=1.3.1' pytest
RUN git reset --hard c0f1fc8d31df45371d5ae6689d3e0a39c058c3ac && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
