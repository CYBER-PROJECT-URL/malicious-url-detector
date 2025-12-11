"""
Backend API for the Malicious URL Detection System
---------------------------------------------------

This service exposes two endpoints:

1. POST /api/v1/scan
   - Receives a URL from the client
   - Sends an asynchronous Celery task to the worker
   - Returns a task_id

2. GET /api/v1/status/{task_id}
   - Fetches the task status and result from Celery
   - Returns prediction results once completed

This file must stay fully synchronized with:
- worker.py (task name + communication format)
- frontend/script.js (response JSON structure)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from celery import Celery
import os

# -------------------------------------------------------------
# Celery Configuration
# -------------------------------------------------------------
# IMPORTANT:
# Inside Docker, Redis is NOT "localhost", it is the service name: "redis"
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')

celery_app = Celery(
    'tasks',
    broker=f'redis://{REDIS_HOST}:6379/0',
    backend=f'redis://{REDIS_HOST}:6379/0'
)

# -------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------
app = FastAPI(
    title="Malicious URL Detector API",
    description="Asynchronous API for real-time URL classification using ML models.",
    version="1.0.0"
)


# -------------------------------------------------------------
# Request Model
# -------------------------------------------------------------
class URLRequest(BaseModel):
    url: str


# -------------------------------------------------------------
# Helper: sanitize URL
# -------------------------------------------------------------
def normalize_url(url: str) -> str:
    """
    Normalize URL format (same logic as inference pipeline).
    Ensures consistency between backend and worker.
    """
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    return url.lower()


# -------------------------------------------------------------
# API Endpoint: Submit URL for analysis
# -------------------------------------------------------------
@app.post("/api/v1/scan", response_model=Dict[str, Any], summary="Scan a URL for malicious behavior")
async def scan_url(request: URLRequest):
    """
    Accepts a URL string and sends an asynchronous Celery task.
    Returns a task_id that the client can later query.
    """

    if not request.url:
        raise HTTPException(status_code=400, detail="URL must be provided")

    clean_url = normalize_url(request.url)

    # TASK NAME MUST MATCH worker.py EXACTLY
    task = celery_app.send_task("analyze_url_for_malware", args=[clean_url])

    return {
        "status": "Processing",
        "task_id": task.id,
        "submitted_url": clean_url,
        "message": "URL submitted successfully. Use /status/{task_id} to check results."
    }


# -------------------------------------------------------------
# API Endpoint: Check task result
# -------------------------------------------------------------
@app.get("/api/v1/status/{task_id}", response_model=Dict[str, Any], summary="Retrieve scan result")
async def get_scan_status(task_id: str):
    """
    Checks the status of the Celery task.
    If completed, returns the prediction & score from worker.py.
    """

    task = celery_app.AsyncResult(task_id)

    if task.state == "PENDING":
        return {
            "status": "Pending",
            "message": "Task is waiting in the queue."
        }

    if task.state == "FAILURE":
        return {
            "status": "Failed",
            "error": str(task.result),
            "message": "Task failed due to an internal error."
        }

    if task.state != "SUCCESS":
        return {
            "status": task.state,
            "message": f"Task currently in state: {task.state}"
        }

    # SUCCESS → return worker results
    return {
        "status": "Completed",
        "result": task.result,
        "message": "URL analysis finished successfully."
    }


# API documentation available at /docs
