FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/scrapy/scrapy.git /testbed/
WORKDIR /testbed/
RUN uv venv --python 3.9
RUN uv pip install 'Twisted>=17.9.0,<22.0.0' 'cryptography>=2.0,<37.0.0' 'cssselect>=0.9.1' 'itemloaders>=1.0.1' 'parsel>=1.5.0' 'pyOpenSSL>=16.2.0,<22.0.0' 'queuelib>=1.4.2' 'service_identity>=16.0.0' 'w3lib>=1.17.0' 'zope.interface>=4.1.3' 'protego>=0.1.15' 'itemadapter>=0.1.0' setuptools tldextract 'lxml>=3.5.0' 'PyDispatcher>=2.0.5' attrs pytest pytest-cov==3.0.0 pytest-xdist 'sybil>=1.3.0' testfixtures
RUN git reset --hard d7b26edf6b419e379a7a0a425093f02cac2fcf33 && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
