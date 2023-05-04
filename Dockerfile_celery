FROM python:3.8.16-bullseye
WORKDIR ..
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
ENTRYPOINT celery -A api worker --loglevel=info
