"""
VibeChecker AI — Seed Database with Sample Data
Run this after init_db.py to populate the database with fake test data.

Usage:
    python seed_db.py
"""

import json
import random
from datetime import datetime, timedelta
from models import get_db, User, Checkin, EmotionResult, SeasonalSummary

# ── Emotion categories (matches FER2013) ────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def random_scores():
    """Generate a random emotion score breakdown that sums to ~1.0."""
    raw = [random.random() for _ in EMOTIONS]
    total = sum(raw)
    return {emotion: round(score / total, 3) for emotion, score in zip(EMOTIONS, raw)}


def get_season(month):
    """Determine season from month number."""
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"


def seed():
    db = get_db()

    try:
        # ── Create sample users ─────────────────────────────
        users = [
            User(
                username="Javaya",
                email="javaya@example.com",
                password_hash="hashed_password_placeholder_1",  # In production, use bcrypt
                timezone="America/Los_Angeles",
            ),
            User(
                username="TestUser",
                email="testuser@example.com",
                password_hash="hashed_password_placeholder_2",
                timezone="America/New_York",
            ),
        ]
        db.add_all(users)
        db.commit()
        print(f"Created {len(users)} users")

        # ── Create check-ins for winter 2026 (Dec 2025 - Feb 2026) ──
        start_date = datetime(2025, 12, 1)
        checkin_count = 0

        for user in users:
            for day_offset in range(90):  # ~3 months of daily check-ins
                date = start_date + timedelta(days=day_offset)

                # Skip some days randomly (realistic — users won't check in every day)
                if random.random() < 0.15:
                    continue

                scores = random_scores()
                predicted = max(scores, key=scores.get)

                checkin = Checkin(
                    user_id=user.user_id,
                    image_path=f"storage/images/{user.user_id}/{date.strftime('%Y-%m-%d')}.jpg",
                    captured_at=date.isoformat(),
                    season=get_season(date.month),
                    season_year=2026 if date.month <= 2 else 2025,
                )
                db.add(checkin)
                db.flush()  # Get the checkin_id before creating the emotion result

                result = EmotionResult(
                    checkin_id=checkin.checkin_id,
                    predicted_emotion=predicted,
                    confidence=round(scores[predicted], 3),
                    scores_json=json.dumps(scores),
                    model_version="v0.1-test",
                )
                db.add(result)
                checkin_count += 1

        db.commit()
        print(f"Created {checkin_count} check-ins with emotion results")

        # ── Create a sample seasonal summary ────────────────
        summary = SeasonalSummary(
            user_id=users[0].user_id,
            season="winter",
            season_year=2026,
            total_checkins=checkin_count // 2,  # Roughly half belong to first user
            avg_happiness=0.18,
            avg_sadness=0.22,
            dominant_emotion="neutral",
            depression_flag=0,
        )
        db.add(summary)
        db.commit()
        print("Created 1 seasonal summary")

        print()
        print("Seed complete! Your database is ready for testing.")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
