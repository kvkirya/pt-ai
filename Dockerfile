FROM python:3.10.6-bullseye

WORKDIR /server

COPY requirements.txt requirements.txt

COPY setup.py setup.py

COPY ptai ptai

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .


CMD uvicorn ptai.api.movenet:app --host 0.0.0.0 --port 8080
