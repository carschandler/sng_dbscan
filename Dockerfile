# FROM ghcr.io/prefix-dev/pixi:latest
FROM rust:latest

WORKDIR /usr/src/myapp

COPY rust .

ENV TZ=Chicago

RUN curl -fsSL https://pixi.sh/install.sh | bash 

RUN /root/.pixi/bin/pixi install --no-progress
RUN /root/.pixi/bin/pixi run maturin develop

RUN bash
