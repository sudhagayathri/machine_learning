FROM python:3.7-buster


RUN apt-get update && apt-get install -y  && apt full-upgrade -y \
    python3-pip \
    gettext-base

WORKDIR /cat-predictor

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY data ./data 
COPY templates ./templates
COPY cat_dog.h5 app.py ./
EXPOSE 5000
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CMD ["python", "./app.py"]