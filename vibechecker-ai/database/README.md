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
from database.db import create_user, get_user_history, store_emotion_result, get_season
```

### Heads up: the `is_latest` flag on `emotion_results`
The `emotion_results` table is **1:many** with `checkins` — a single selfie can
have multiple predictions (one per model version). Only the current one has
`is_latest = 1`; older ones are kept as history with `is_latest = 0`.

- `store_emotion_result()` handles the flag automatically — it demotes any
  existing latest row before inserting the new one. Always go through this
  helper; don't insert into `emotion_results` directly.
- **Any aggregate/dashboard query must filter `EmotionResult.is_latest == 1`**
  or you'll double-count historical predictions. All helpers in `db.py`
  (`get_user_history`, `get_emotion_counts`, `get_average_scores`,
  `get_weekly_sadness_trend`, `get_dominant_emotion`) already do this.
- For the full history of a single check-in (e.g. comparing model versions),
  use `get_emotion_result_history(checkin_id)`.

### Season helper
`get_season(month)` returns `"winter" | "spring" | "summer" | "fall"`. Use it
when creating a check-in so the `season` / `season_year` columns stay
consistent across the codebase:
```python
from datetime import datetime
from database.db import get_season, create_checkin

now = datetime.utcnow()
season = get_season(now.month)
season_year = now.year + 1 if now.month == 12 else now.year  # winter rolls forward
```
