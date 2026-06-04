# Workflow-CI

Repository ini dibuat untuk memenuhi Kriteria 3 submission Membangun Sistem Machine Learning.

## Tujuan

Repository ini menjalankan workflow CI untuk melakukan retraining model machine learning secara otomatis menggunakan MLflow Project.

## Dataset

Dataset yang digunakan adalah hasil preprocessing dari Telco Customer Churn.

## Struktur Repository

```text
Workflow-CI
├── .github
│   └── workflows
│       └── ci.yml
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   ├── requirements.txt
│   └── telco_customer_churn_preprocessing
├── DockerHub.txt
├── latest_run_id.txt
└── mlruns