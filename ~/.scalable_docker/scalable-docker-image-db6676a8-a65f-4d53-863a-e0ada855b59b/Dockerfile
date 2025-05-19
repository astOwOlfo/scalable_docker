FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y wget git build-essential libffi-dev libtiff-dev python3 python3-full python3-pip python3-venv python-is-python3 jq curl locales locales-all tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /testbed/
RUN python -m venv .venv
RUN echo '. .venv/bin/activate' >> ~/.bashrc
RUN .venv/bin/python -m pip install numexpr
RUN mkdir assignment
RUN echo ZnJvbSBzZXR1cHRvb2xzIGltcG9ydCBzZXR1cCwgZmluZF9wYWNrYWdlcwoKc2V0dXAoCiAgICBuYW1lPSJhc3NpZ25tZW50IiwKICAgIHZlcnNpb249IjAuMS4wIiwKICAgIHBhY2thZ2VzPWZpbmRfcGFja2FnZXMoKSwKICAgIGluc3RhbGxfcmVxdWlyZXM9WydudW1leHByJ10sCiAgICBweXRob25fcmVxdWlyZXM9Ij49My45IiwKKQo= | base64 -d > setup.py
