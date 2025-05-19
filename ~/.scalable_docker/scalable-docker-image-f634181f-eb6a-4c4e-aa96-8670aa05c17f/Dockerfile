FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/dask/dask.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.10
RUN uv pip install 'numpy<2.0.0' pytest pytest-rerunfailures pytest-xdist toolz 'cloudpickle>=1.1.1' 'fsspec>=0.6.0' 'packaging>=20.0' 'partd>=0.3.10' 'pyyaml>=5.3.1'
RUN git reset --hard fa6a5d619d42877283042c85abccc620f1ede100 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
