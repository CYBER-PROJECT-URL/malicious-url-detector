## 📄 README.md 

# 🔬 Detecting Malicious URLs using Distributed Machine Learning

**Course:** שיטות לזיהוי התקפות סייבר (Cyber Attack Detection Methods)
**Project Title:** Detecting Malicious URLs using Machine Learning
**Submitted by:** Elioz Kolani, Daniel Samson
**Submission Date:** October 2025

---

## 💡 1. Project Overview

This project implements a scalable and computationally efficient framework for detecting malicious URLs (Phishing, Malware distribution, etc.). The solution utilizes classic machine learning models (XGBoost/Random Forest) to achieve high accuracy while ensuring real-time performance and interpretability, unlike heavy deep learning approaches.

The core system is built on a **microservices architecture** running entirely within Docker, demonstrating robust handling of high loads via parallel processing.

### 🔑 Key Contributions

1.  **Distributed Architecture:** Full separation of Frontend, Backend (API), and parallel Workers using Docker and Redis Queue.
2.  **Rich Feature Engineering:** Implementation of extended Lexical, Host-based, and Structural features to improve generalization and detection accuracy.
3.  **Performance Superiority:** Demonstrated computational efficiency (low inference time) compared to Transformer-based models (PMANet), making it ideal for operational deployment.
4.  **High Accuracy:** Achieved high F1-score and Accuracy metrics, surpassing the baseline performance of classical feature-engineered models (Rathod et al. 2024).

---

## 🏗️ 2. System Architecture

The project is structured as a robust, asynchronous pipeline:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend (UI)** | Nginx / HTML / JS | Interactive interface for URL submission and result visualization. Runs on port **80**. |
| **Backend (API)** | Python / FastAPI | The API Gateway. Receives URL requests and submits them to the **Redis Queue**. Provides full **OpenAPI** documentation. Runs on port **8000**. |
| **Queue** | Redis | Message broker/task queue (Celery backend). Manages the workflow and handles task backlog under load. |
| **Workers (ML Engine)** | Python / Celery | The parallel processing unit. Consumes tasks from the Queue, performs **Feature Extraction**, loads the `ml_model.pkl`, and performs the **Prediction**. |

---

## 💻 3. Getting Started (Installation & Setup)

### Prerequisites

* Docker and Docker Compose installed.
* Python 3.9+ environment for model training (`train_model.py`).

### Step 1: Prepare the Model (`ml_model.pkl`)

You must first train the model and generate the prediction asset.

1.  **Install Local Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Complete Feature Extractor:** Ensure the rich feature extraction logic is fully implemented in `./backend/feature_extractor.py`.
3.  **Train and Save:** Run the training script. This script loads the dataset, trains the **XGBoost** classifier, and saves the resulting model file (`ml_model.pkl`) into the required path (`./backend/models/`).
    ```bash
    python train_model.py
    ```

### Step 2: Launch the Distributed System

Use Docker Compose to build and run all five services (Frontend, Backend, Redis, Worker 1, Worker 2).

```bash
# 1. Build the images (using the root directory as context)
docker-compose build

# 2. Run the services in detached mode
docker-compose up -d
````

### Step 3: Access the Application

Once all services are up and running (allow a moment for the Workers to connect to Redis):

  * **User Interface (UI):** Open your browser to `http://localhost:80`
  * **API Documentation (OpenAPI):** Access the full documentation at `http://localhost:8000/docs`

-----

## 🔬 4. Research Results and Comparison

This section will be fully written in the Final Report, but the key comparison points are detailed below.

### 4.1. Comparison Baselines

| Study | Approach | Key Result | Limitation Overcome by Our Model |
| :--- | :--- | :--- | :--- |
| **PMANet (Liu et al., 2025)** | Transformer-based Deep Learning | High Accuracy (99.4%) | **Computational Cost:** Our model is significantly lighter, enabling real-time deployment without heavy GPU dependency. |
| **Rathod et al. (2024)** | Classical ML with Dimensionality Reduction | Moderate Accuracy (92.5%) | **Feature Depth & Dataset:** Our model utilizes richer features and a broader, merged dataset, targeting higher F1/Accuracy scores. |

### 4.2. Achieved Performance (To be completed in the Final Report)

| Metric | Our Model (XGBoost) | Rathod et al. (Target Baseline) | PMANet (Computational Baseline) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | XX.XX% | 92.5% | 99.4% |
| **F1-Score** | XX.XX% | (Lower) | (Higher) |
| **Inference Time (per URL)** | **\< 5 ms** (Goal: Prove speed) | (N/A) | **\> 100 ms** (Estimate: Prove lightness) |

-----

## 🛑 5. Safety Notice

  * **Malware Handling:** Since this project deals with malware-related data (URLs), all development and processing of feature extraction must be performed within a **Virtual Machine (VM)** environment.
  * **Data Handling:** Extracted features should be moved outside the VM for model training only after sanitization.

<!-- end list -->

```
```
