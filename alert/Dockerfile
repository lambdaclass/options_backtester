FROM debian:latest
MAINTAINER Amin Arria <amin.arria@lambdaclass.com>

WORKDIR /root
RUN apt-get update
RUN apt-get install -y curl make build-essential pkg-config openssl libssl-dev

RUN apt-get -y install firefox-esr
RUN curl -L -O  'https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-linux64.tar.gz'
RUN tar -vxf geckodriver-v0.24.0-linux64.tar.gz

RUN apt-get install -y sqlite3 libsqlite3-dev

RUN curl https://sh.rustup.rs -sSf > rustup
RUN sh rustup -y
ENV PATH=/root/.cargo/bin:$PATH

COPY . watchdog/
WORKDIR watchdog
RUN make install_diesel_cli
RUN make migration
RUN make release

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
CMD (/root/geckodriver &) && ./target/release/alert
