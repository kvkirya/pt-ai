# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.10.6-buster

WORKDIR /ptai-proj

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ptai ptai
COPY setup.py setup.py
RUN pip install .

COPY raw_data/models raw_data/models

# COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn ptai.api.fast:app --host 0.0.0.0 --port $PORT
