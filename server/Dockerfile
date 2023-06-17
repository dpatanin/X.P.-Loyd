FROM python:3.9

WORKDIR /app

COPY ./server .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./lib/ /app/lib
EXPOSE 8000

CMD [ "python", "server.py" ]