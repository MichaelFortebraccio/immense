FROM python:3.8.16-bullseye
RUN apt-get update
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN mkdir "jobs"
COPY . /counter_sairus
WORKDIR /counter_sairus
EXPOSE 5000
ENTRYPOINT ["python", "api.py"]








