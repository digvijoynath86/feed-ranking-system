# Feed Ranking System using LightGBM, MLflow, FastAPI, Docker & AWS

## Overview

This project implements a production-style Feed Ranking System inspired by real-world recommendation pipelines used in platforms

The system generates personalized feeds for users by ranking candidate tweets/posts using machine learning models trained on user engagement signals.

The project demonstrates:

- End-to-end ML pipeline development
- Feature engineering for ranking systems
- Learning-to-Rank using LightGBM Ranker
- Experiment tracking using MLflow
- Data versioning using DVC
- API serving using FastAPI
- Docker-based deployment
- AWS EC2 deployment

---

# System Architecture

```text
User Request
     ↓
FastAPI Service
     ↓
Feature Loader
     ↓
MLflow Model Loader
     ↓
LightGBM Ranking Model
     ↓
Ranked Feed Response
```

---

# Project Structure

```text
feed-ranking/
│
├── app/
│   ├── main.py
│   ├── service.py
│   └── schemas.py
│
├── pipeline/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── reranker.py
│
├── stages/
│   ├── stage_01_data_generation.py
│   ├── stage_02_feature_engineering.py
│   ├── stage_03_train.py
│   └── stage_04_evaluation.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── mlruns/
│
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

# Key Features Used

## User-Author Interaction Features

- engagement_3m
- engagement_1y

## Author Features

- is_verified
- num_followers
- social_rank

## User-Tweet Features

- topic_similarity

## Tweet Features

- tweet_length
- recency_hours
- has_media
- is_url

## Contextual Features

- day_of_week
- time_of_day
- holiday_approaching

---

# Ranking Model

## Model Used

```text
LightGBM Ranker (LGBMRanker)
```

## Why Learning-to-Rank?

Unlike classification models, ranking models optimize:

```text
relative ordering of content
```

instead of predicting only:

```text
engagement probability
```

This closely matches real-world feed ranking systems.

---

# Evaluation Metrics

## NDCG@10

Normalized Discounted Cumulative Gain evaluates:

- relevance of ranked items
- position sensitivity
- ranking quality

Higher ranked relevant tweets receive higher scores.

## Additional Feed Metrics

- Precision@10
- Diversity
- Author Diversity
- Recency

---

# A/B Testing Simulation

The project includes:

- Baseline model (Model A)
- Improved ranking model (Model B)
- Feed-level comparison
- Offline experimentation

This simulates real-world experimentation pipelines used in large-scale recommendation systems.

---

# MLflow Integration

MLflow is used for:

- Experiment tracking
- Metric logging
- Model artifact storage
- Model loading during inference

## Logged Artifacts

- LightGBM model
- NDCG metrics
- Feature importance
- Parameters

---

# DVC Integration

DVC is used for:

- Pipeline orchestration
- Data versioning
- Reproducibility

## Example

```bash
dvc repro
```

This executes the complete ML pipeline.

---

# FastAPI Serving

The project exposes a REST API for real-time feed generation.

## Run API Locally

```bash
uvicorn app.main:app --reload
```

## Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# Example API Request

## Endpoint

```text
POST /feed
```

## Request

```json
{
  "user_id": "U1"
}
```

## Response

```json
{
  "user_id": "U1",
  "feed": [
    {
      "tweet_id": "T23",
      "score": 0.91
    },
    {
      "tweet_id": "T11",
      "score": 0.87
    }
  ]
}
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t feed-ranking-api .
```

## Run Container

```bash
docker run -d -p 8000:8000 feed-ranking-api
```

---

# AWS EC2 Deployment

The application was deployed on AWS EC2 using Docker.

## Deployment Stack

```text
AWS EC2
   ↓
Docker Container
   ↓
FastAPI
   ↓
MLflow Model
```

## Steps Performed

- Created IAM user
- Launched EC2 instance
- Configured security groups
- Installed Docker
- Copied project to EC2
- Built Docker image
- Exposed FastAPI endpoint publicly

---

# Local Setup

## Clone Repository

```bash
git clone <repository-url>
cd feed-ranking
```

## Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Training Pipeline

## Execute Full Pipeline

```bash
dvc repro
```

---

# Run Individual Stages

## Data Generation

```bash
python stages/stage_01_data_generation.py
```

## Feature Engineering

```bash
python stages/stage_02_feature_engineering.py
```

## Training

```bash
python stages/stage_03_train.py
```

## Evaluation

```bash
python stages/stage_04_evaluation.py
```

---

# Technologies Used

| Category | Technology |
|---|---|
| ML Model | LightGBM |
| API | FastAPI |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker |
| Cloud | AWS EC2 |
| Language | Python |
| Data Processing | Pandas, NumPy |

---

