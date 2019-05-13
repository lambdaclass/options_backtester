FROM debian:latest
MAINTAINER Juan Pablo Amoroso <jamoroso@lambdaclass.com>

RUN apt-get update
RUN apt-get install -y python3 python3-pip make cron build-essential pkg-config openssl libssl-dev
RUN python3 -m pip install pipenv
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

COPY . /finance
WORKDIR /finance

RUN make init

COPY ./docker/data_scraper/crontab /etc/cron.d/scraper-cron
COPY ./docker/data_scraper/entrypoint.sh /usr/bin/entrypoint.sh
COPY ./docker/data_scraper/run-task.sh /usr/bin/run-task
RUN chmod 0644 /etc/cron.d/scraper-cron && crontab /etc/cron.d/scraper-cron
ENTRYPOINT ["entrypoint.sh"]
CMD ["cron", "-f"]
