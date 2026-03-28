FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../src ./src

EXPOSE 7860
CMD ["python", "app.py"]
