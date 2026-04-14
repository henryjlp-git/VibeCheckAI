# Database / Storage — Javaya

Responsible for:
- Schema design (SQLite + SQLAlchemy)
- Image / result storage
- Query support for trends and history

## Files
- `models.py` — SQLAlchemy table definitions (User, Checkin, EmotionResult, SeasonalSummary)
- `init_db.py` — Run once to create the database
- `seed_db.py` — Populate with fake test data
- `db.py` — Helper functions for the backend to import

## Setup
```bash
cd database
python init_db.py      # Creates vibechecker.db
python seed_db.py      # Seeds with test data
```

## For Backend (Zem)
```python
from database.db import create_user, get_user_history, store_emotion_result
```
