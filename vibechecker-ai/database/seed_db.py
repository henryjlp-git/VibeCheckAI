"""Populate vibechecker.db with fake users and check-ins for testing."""

import json
import random
from datetime import datetime, timedelta
from models import get_db, User, Checkin, EmotionResult, SeasonalSummary
from db import get_season

EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]


def random_scores():
    """Random normalized emotion scores that sum to ~1.0."""
    raw = [random.random() for _ in EMOTIONS]
    total = sum(raw)
    return {emotion: round(score / total, 3) for emotion, score in zip(EMOTIONS, raw)}


def seed():
    db = get_db()
    try:
        users = [
            User(
                username="Javaya",
                email="javaya@example.com",
                password_hash="hashed_password_placeholder_1",  # use bcrypt in real code
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

        # Winter 2026 = Dec 2025 through Feb 2026
        start_date = datetime(2025, 12, 1)
        checkin_count = 0

        for user in users:
            for day_offset in range(90):
                date = start_date + timedelta(days=day_offset)

                # skip ~15% of days so it isn't perfectly daily
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
                db.flush()  # need checkin_id for the FK below

                result = EmotionResult(
                    checkin_id=checkin.checkin_id,
                    predicted_emotion=predicted,
                    confidence=round(scores[predicted], 3),
                    scores_json=json.dumps(scores),
                    model_version="v0.1-test",
                    is_latest=1,
                )
                db.add(result)
                checkin_count += 1

        db.commit()
        print(f"Created {checkin_count} check-ins with emotion results")

        summary = SeasonalSummary(
            user_id=users[0].user_id,
            season="winter",
            season_year=2026,
            total_checkins=checkin_count // 2,
            avg_happiness=0.18,
            avg_sadness=0.22,
            dominant_emotion="neutral",
            depression_flag=0,
        )
        db.add(summary)
        db.commit()
        print("Created 1 seasonal summary")
        print()
        print("Seed complete.")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
