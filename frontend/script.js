// frontend/script.js

// NOTE: The backend service is named 'backend' in Docker Compose, accessible on port 8000.
// We use the mapped port for ease of access from the host machine:
const API_URL = "http://localhost:8000/api/v1";

async function scanUrl() {
    const url = document.getElementById('urlInput').value;
    if (!url) return alert('Please enter a URL.');

    document.getElementById('taskStatus').className = 'pending';
    document.getElementById('finalResult').innerHTML = '';

    // 1. Submit the URL to the Backend (API Gateway)
    const response = await fetch(`${API_URL}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
    });

    if (!response.ok) {
        document.getElementById('taskStatus').innerHTML = `<span class="malicious">Submission Error: ${response.status}</span>`;
        return;
    }

    const data = await response.json();
    const taskId = data.task_id;
    document.getElementById('taskStatus').innerText = `Task submitted. ID: ${taskId}. Waiting for Worker pickup...`;

    // 2. Start polling for status
    const checkStatusInterval = setInterval(async () => {
        const statusResponse = await fetch(`${API_URL}/status/${taskId}`);
        const statusData = await statusResponse.json();

        // Update status display
        document.getElementById('taskStatus').innerText = `Current Status: ${statusData.status}`;

        if (statusData.status === 'Completed') {
            clearInterval(checkStatusInterval);

            const result = statusData.result;
            let resultText;
            let resultClass;

            if (result.is_malicious) {
                resultText = 'MALICIOUS';
                resultClass = 'malicious';
            } else {
                resultText = 'BENIGN';
                resultClass = 'benign';
            }

            document.getElementById('finalResult').innerHTML = `
                <p><strong>URL:</strong> ${result.url}</p>
                <p><strong>Prediction:</strong> <span class="${resultClass}">${resultText}</span></p>
                <p><strong>Confidence:</strong> ${result.confidence} (${result.model_type})</p>
                <p>This result was processed by one of the parallel Workers.</p>
            `;
        } else if (statusData.status === 'Failed') {
            clearInterval(checkStatusInterval);
            document.getElementById('finalResult').innerHTML = `<span class="malicious">Processing FAILED: ${statusData.result}</span>`;
        }
    }, 3000); // Poll every 3 seconds
}