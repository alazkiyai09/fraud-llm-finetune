FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY inference/requirements.txt /app/inference/requirements.txt
RUN pip install --no-cache-dir -r /app/inference/requirements.txt

COPY inference /app/inference
COPY src /app/src

EXPOSE 8080

CMD ["sh", "-c", "python inference/app.py"]
