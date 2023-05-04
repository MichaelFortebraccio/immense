FROM python:3.8.16-bullseye
RUN apt-get update
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
WORKDIR /counter_sairus
RUN mkdir "jobs"
COPY . /counter_sairus
EXPOSE 5000
RUN celery -A api worker --loglevel=info
ENTRYPOINT ["python", "api.py"]








