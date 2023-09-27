FROM python:3.9

WORKDIR /app

COPY ./server .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD [ "python", "server.py" ]