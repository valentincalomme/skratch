FROM python:3.6.4-alpine3.7 as env

ENV ALPINE_VERSION 3.7

RUN apk update &&\
    apk upgrade &&\
    apk add sudo g++




CMD ["python", "-c" "print('hello world')"]
