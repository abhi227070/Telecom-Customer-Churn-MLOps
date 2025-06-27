# 📊 Telecom Customer Churn Prediction - MLOps Implementation

This project demonstrates a complete **MLOps pipeline** for predicting customer churn in a telecom dataset. While the model itself is relatively simple, the emphasis is on production-grade architecture, CI/CD, and deployment using best MLOps practices.

---

## 📚 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🧰 Tech Stack](#-tech-stack)
- [🗂️ Folder Structure](#️-folder-structure)
- [🔁 MLOps Pipeline Stages](#-mlops-pipeline-stages)
- [🐳 Docker Setup](#-docker-setup)
- [🧪 API Endpoints (FastAPI)](#-api-endpoints-fastapi)
- [🧾 Logging & Monitoring](#-logging--monitoring)
- [⚙️ CI/CD with GitHub Actions](#️-cicd-with-github-actions)
- [🧪 Testing](#-testing)
- [📦 Setup Locally](#-setup-locally)
- [📈 Future Work](#-future-work)
- [🙌 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

---

## 🚀 Project Overview

- **Problem**: Predict whether a customer will churn based on their telecom usage.
- **Goal**: Build an end-to-end MLOps pipeline that includes data ingestion, validation, model training, evaluation, deployment, and monitoring.
- **Deployment**: AWS EC2 (Dockerized model API)
- **CI/CD**: GitHub Actions or your preferred CI/CD tool
- **Database**: MongoDB for storing raw and processed data

---

## 🧰 Tech Stack

| Component          | Technology           |
|-------------------|----------------------|
| Language           | Python               |
| Framework          | FastAPI / Flask      |
| ML Model           | Scikit-learn         |
| CI/CD              | GitHub Actions       |
| Containerization   | Docker               |
| Cloud Deployment   | AWS EC2              |
| Data Storage       | MongoDB              |
| Orchestration      | Custom Python pipeline |
| Logging            | Python logging module |
| Configuration      | `dataclasses`, `dotenv`, constants file |

---