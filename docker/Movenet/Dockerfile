# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.10.6

WORKDIR /pt-ai-project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ptai ptai
COPY setup.py setup.py
RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
