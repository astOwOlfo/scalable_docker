FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/PrefectHQ/prefect.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install 'click>=7.0,<8.0' 'cloudpickle>=0.6.0,<1.3' 'croniter>=0.3.24,<0.3.31' 'dask[bag]>=0.19.3,<3.0' 'distributed>=1.26.1,<3.0' 'docker>=3.4.1,<5.0' 'marshmallow>=3.0.0b19,<=3.2.1' 'marshmallow-oneofschema>=2.0.0b2,<3.0' 'mypy_extensions>=0.4.0,<1.0' 'pendulum>=2.0.4,<3.0' 'python-dateutil~=2.7' 'pyyaml>=3.13,<5.4' 'python-box>=3.4.4,<5.0' 'python-slugify>=1.2.6,<5.0' 'pytz>=2018.7' 'requests>=2.20,<3.0' 'supervisor>=4.0,<5.0' 'tabulate>=0.8.0,<1.0' 'toml>=0.9.4,<1.0' 'typing>=3.6.1,<4.0' 'typing_extensions>=3.6.2,<4.0' 'urllib3>=1.24.3' 'pytest>=5.0,<6.0' 'testfixtures>=6.10.3,<7.0' 'pytest-cov>=2.6,<3.0' 'pytest-env>=0.6.0,<0.7.0' 'pytest-xdist>=1.23,<2.0'
RUN git reset --hard 87b762d561ad4be2d7c8dec97f23cf02722fbb7b && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
