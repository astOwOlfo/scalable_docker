FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/joke2k/faker.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.8
RUN uv pip install mock==2.0.0 UkPostcodeParser==1.1.2 email_validator==1.0.3 'python-dateutil>=2.4' 'six>=1.10' text-unidecode==1.2 pytest
RUN git reset --hard 434457ea74daf0d9c3c7380eb91af4ae34936725 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
