FROM tensorflow/serving

WORKDIR /

COPY . /server

ENV MODEL_NAME="prototype-V1_terminal_19_05_2023 11_23_45.pb"
ENV MODEL_BASE_PATH=/

EXPOSE 8501
RUN apt-get -y update && apt-get install -y python

CMD ["python", "server.py"]