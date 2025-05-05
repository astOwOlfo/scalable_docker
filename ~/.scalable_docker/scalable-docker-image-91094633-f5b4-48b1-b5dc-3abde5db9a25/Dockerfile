
FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata

# RUN apt update && apt install -y libpq-dev python3-dev freetds-dev libkrb5-dev libmpich-dev

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# Download and install conda
# RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh' -O miniconda.sh #     && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
# ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
# RUN conda init --all
# RUN conda config --append channels conda-forge

# RUN git clone https://github.com/modin-project/modin.git /testbed/
# WORKDIR /testbed/
# RUN git checkout 21ab814e2f9fd9e4874a036e2fa8e53208638614

RUN git clone https://github.com/mesonbuild/meson.git /testbed/
WORKDIR /testbed/
RUN git checkout 73c06780f93f7565a39def7ed08e404fbcaae358
