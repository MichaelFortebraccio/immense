from celery import Celery
import os

BROKER_URI = os.environ["BROKER_URI"]
BACKEND_URI = os.environ["BACKEND_URI"]

# celery -A task_manager/worker.celery worker --loglevel=info
celery = Celery('worker', backend=BACKEND_URI, broker=BROKER_URI, include=['task_manager.tasks'])
