FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../src ./src
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app

EXPOSE 7860
USER appuser
CMD ["python", "app.py"]
