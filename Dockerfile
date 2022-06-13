FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./load_tfdata.py .
