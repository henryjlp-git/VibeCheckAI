# Backend — Zem

Responsible for:
- API routes (check-in upload, user auth, history retrieval)
- Inference endpoints (send image to model, get emotion back)
- Server logic

## Database Access
Import helper functions from the database module:
```python
from database.db import create_user, create_checkin, store_emotion_result, get_user_history
```

## Getting Started
TBD — add setup instructions once the framework is chosen (Flask / FastAPI / etc).
