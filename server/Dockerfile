FROM tensorflow/serving:latest-gpu

WORKDIR /

COPY . /server

ENV MODEL_BASE_PATH=/server

EXPOSE 8501
RUN apt-get -y update && apt-get install -y python

CMD ["python", "server.py"]