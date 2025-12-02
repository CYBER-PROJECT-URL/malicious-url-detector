# worker/Dockerfile
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt from the root directory and install dependencies
# Note the change: copying from ../requirements.txt to .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY worker.py .

# Command to run the worker is defined in docker-compose.yaml