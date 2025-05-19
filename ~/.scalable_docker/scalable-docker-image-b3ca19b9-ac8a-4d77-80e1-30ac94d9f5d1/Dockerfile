FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/openvinotoolkit/datumaro.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install 'attrs>=21.1.0' 'defusedxml>=0.7.0' 'lxml>=4.4.1' 'matplotlib>=3.3.1' 'networkx>=2.5' 'numpy>=1.17.3' 'Pillow>=6.1.0' 'ruamel.yaml>=0.17.0' 'typing_extensions>=3.7.4.3' 'pycocotools>=2.0.4' 'PyYAML>=5.3.1' 'tensorboardX>=1.8,!=2.3' scipy 'pandas>=1.1.5' 'pytest>=5.3.5' 'opencv-python-headless>=4.1.0.25'
RUN git reset --hard 8a65a9072cd0399adf4fa7ba37390bb1f5ed027e && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
