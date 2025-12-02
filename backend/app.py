# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
from celery import Celery

# Configuration for Celery (using Redis as the broker/backend)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
celery_app = Celery('tasks', broker=f'redis://{REDIS_HOST}:6379/0', backend=f'redis://{REDIS_HOST}:6379/0')
# Note: The backend is used to store and retrieve task results

app = FastAPI(
    title="Malicious URL Detector API",
    description="API for real-time URL classification using a lightweight, distributed ML model.",
    version="1.0.0"
)  # Ensures full OpenAPI documentation


class URLRequest(BaseModel):
    url: str


@app.post("/api/v1/scan", response_model=Dict[str, Any], summary="Scan a URL for Malicious Content")
async def scan_url(request: URLRequest):
    """
    Receives a URL and submits an asynchronous task to the Celery Workers for processing.
    """
    if not request.url:
        raise HTTPException(status_code=400, detail="URL must be provided")

    # Send the task to the queue (the 'worker' service will pick it up)
    # The task name must match the one defined in worker/worker.py
    task = celery_app.send_task('worker.task_process_url', args=[request.url])

    return {
        "status": "Processing",
        "task_id": task.id,
        "message": "URL submitted for analysis. Check the status endpoint for the result."
    }


@app.get("/api/v1/status/{task_id}", response_model=Dict[str, Any], summary="Get Scan Status and Result")
async def get_scan_status(task_id: str):
    """
    Checks the status of a submitted task and returns the final prediction if completed.
    """
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {"status": "Pending", "message": "Task is waiting in the queue."}
    elif task.state == 'FAILURE':
        response = {"status": "Failed", "result": str(task.result),
                    "message": "Processing failed due to an internal error."}
    elif task.state != 'SUCCESS':
        # States like STARTED, RETRY, etc.
        response = {"status": task.state, "message": f"Task is currently in state: {task.state}"}
    else:
        # Task is successful, return the prediction result
        response = {
            "status": "Completed",
            "result": task.result,
            "message": "Analysis completed."
        }
    return response

# Note: The automatically generated OpenAPI documentation is available at /docs