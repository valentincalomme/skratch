FROM python:3.6.4-alpine3.7 as env

ENV ALPINE_VERSION 3.7

RUN apk update &&\
    apk upgrade &&\
    apk add git

RUN git clone git://github.com/valentincalomme/skratch /src

WORKDIR src/skratch

RUN pip install -r requirements

ADD notebooks/ /home/jovyan/