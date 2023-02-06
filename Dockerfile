FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
