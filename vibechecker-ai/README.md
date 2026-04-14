# VibeChecker AI

Facial emotion tracking app that determines seasonal depression patterns by analyzing daily selfies over the course of a season.

## Tech Stack

- **Frontend:** TBD (vexonik + Nick)
- **Backend:** TBD (Zem)
- **Database:** SQLite + SQLAlchemy (Javaya)
- **Computer Vision:** MediaPipe (Aaron)
- **ML Model:** TBD (Henry + Aaron)
- **Dataset:** FER2013 (Javaya)

## Project Structure

```
vibechecker-ai/
├── frontend/          # UI — daily upload, dashboard, charts (vexonik + Nick)
├── backend/           # API routes, inference endpoints, server logic (Zem)
│   ├── routes/
│   └── services/
├── database/          # Schema, models, DB helpers (Javaya)
├── storage/images/    # User selfies stored here (not committed to git)
├── ml/                # Model training, checkpoints, experiments (Henry + Aaron)
│   ├── models/
│   ├── experiments/
│   └── scripts/
├── data/              # FER2013 dataset + processed data (Javaya)
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
├── cv/                # MediaPipe face mesh + preprocessing (Aaron)
├── notebooks/         # Jupyter notebooks for exploration
└── docs/              # Project documentation
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-team/vibechecker-ai.git
cd vibechecker-ai

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Initialize the database
cd database
python init_db.py

# 4. (Optional) Seed with test data
python seed_db.py
```

## Dataset

Download FER2013 from Kaggle: https://www.kaggle.com/datasets/deadskull7/fer2013
Place the CSV in `data/raw/`.

## Team

| Role | Lead |
|------|------|
| Frontend | vexonik + Nick |
| Backend | Zem |
| Database / Storage | Javaya |
| Computer Vision | Aaron |
| ML Training | Henry + Aaron |
| Data / ML Ops | Javaya |
| Full-stack Integration | TBD |
