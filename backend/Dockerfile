# backend/Dockerfile
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt from the root directory and install dependencies
# Note the change: copying from ../requirements.txt to .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY feature_extractor.py .
RUN mkdir -p models

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]