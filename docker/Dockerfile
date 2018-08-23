FROM python:3.6.2 as env

RUN apt-get update && apt-get install gcc
RUN git clone git://github.com/valentincalomme/skratch /src

WORKDIR src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD notebooks/ /home/jovyan/