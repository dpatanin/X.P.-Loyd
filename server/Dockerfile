FROM python:3.9

WORKDIR /app

COPY ./server .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./lib/ /app/lib
COPY ./config.yaml /app/config.yaml
EXPOSE 8000

CMD [ "python", "server.py" ]