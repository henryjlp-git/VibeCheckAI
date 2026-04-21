"""Database helpers the backend imports.

Usage:
    from db import create_user, create_checkin, get_user_history, ...

All functions that fetch or create records return ORM objects.
Call .to_dict() on any object to get a JSON-safe dict for API responses.
"""

import json
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from models import get_db, User, Checkin, EmotionResult, SeasonalSummary, now_iso

DEPRESSION_THRESHOLD = 0.3  # avg sadness above this triggers depression_flag = 1


# ── Season helper ───────────────────────────────────────────

def get_season(month: int) -> str:
    """Map a month number (1–12) to a season name."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


# ── Users ───────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str, tz: str = "UTC") -> User:
    """Create a user. password_hash must be a bcrypt hash (starts with '$2b$')."""
    if not password_hash.startswith("$2b$"):
        raise ValueError(
            "password_hash must be a bcrypt hash. "
            "Use bcrypt.hashpw(password.encode(), bcrypt.gensalt()) before calling create_user()."
        )
    db = get_db()
    try:
        user = User(username=username, email=email, password_hash=password_hash, timezone=tz)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def get_user_by_email(email: str) -> User | None:
    db = get_db()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def get_user_by_id(user_id: int) -> User | None:
    db = get_db()
    try:
        return db.query(User).filter(User.user_id == user_id).first()
    finally:
        db.close()


# ── Check-ins ───────────────────────────────────────────────

def create_checkin(user_id: int, image_path: str, captured_at: str,
                   season: str, season_year: int) -> Checkin:
    """Record a daily selfie upload."""
    db = get_db()
    try:
        checkin = Checkin(
            user_id=user_id,
            image_path=image_path,
            captured_at=captured_at,
            season=season,
            season_year=season_year,
        )
        db.add(checkin)
        db.commit()
        db.refresh(checkin)
        return checkin
    finally:
        db.close()


# ── Emotion results ─────────────────────────────────────────

def store_emotion_result(checkin_id: int, predicted_emotion: str,
                         confidence: float, scores: dict,
                         model_version: str = "v1.0") -> EmotionResult:
    """Save a new prediction for a check-in.
    If a previous prediction exists, it is demoted to is_latest=0 (kept as history)
    and the new one is inserted with is_latest=1."""
    db = get_db()
    try:
        # Demote any currently-latest result for this checkin.
        (
            db.query(EmotionResult)
            .filter(
                EmotionResult.checkin_id == checkin_id,
                EmotionResult.is_latest == 1,
            )
            .update({"is_latest": 0})
        )

        result = EmotionResult(
            checkin_id=checkin_id,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            scores_json=json.dumps(scores),
            model_version=model_version,
            is_latest=1,
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result
    finally:
        db.close()


def get_emotion_result_history(checkin_id: int) -> list[EmotionResult]:
    """All predictions ever made for one check-in, newest first.
    Useful for comparing what different model versions predicted on the same selfie."""
    db = get_db()
    try:
        return (
            db.query(EmotionResult)
            .filter(EmotionResult.checkin_id == checkin_id)
            .order_by(EmotionResult.processed_at.desc())
            .all()
        )
    finally:
        db.close()


# ── Queries for dashboard + history ─────────────────────────

def get_user_history(user_id: int, season: str, season_year: int) -> list[Checkin]:
    """All check-ins for a user in one season with latest_result eager-loaded.
    Checkins without any prediction yet will have .latest_result = None."""
    db = get_db()
    try:
        return (
            db.query(Checkin)
            .options(joinedload(Checkin.latest_result))
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
            )
            .order_by(Checkin.captured_at.asc())
            .all()
        )
    finally:
        db.close()


def get_weekly_sadness_trend(user_id: int, season: str, season_year: int) -> list[dict]:
    """Week-by-week average sadness for a season. Used for frontend trend charts.

    Returns: [{"week": "2026-01", "avg_sadness": 0.23}, ...]
    Week format is YYYY-WW (ISO year + week number).
    Only the latest prediction per check-in is counted.
    """
    db = get_db()
    try:
        week_col = func.strftime("%Y-%W", Checkin.captured_at).label("week")
        avg_sad_col = func.avg(
            func.json_extract(EmotionResult.scores_json, "$.sad")
        ).label("avg_sadness")

        results = (
            db.query(week_col, avg_sad_col)
            .join(EmotionResult, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
                EmotionResult.is_latest == 1,
            )
            .group_by(week_col)
            .order_by(week_col.asc())
            .all()
        )

        return [
            {"week": row.week, "avg_sadness": round(row.avg_sadness, 4)}
            for row in results
        ]
    finally:
        db.close()


def get_emotion_counts(user_id: int, season: str, season_year: int) -> dict:
    """How many times each emotion appeared in a season (latest predictions only)."""
    db = get_db()
    try:
        results = (
            db.query(EmotionResult.predicted_emotion)
            .join(Checkin, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
                EmotionResult.is_latest == 1,
            )
            .all()
        )

        counts = {}
        for (emotion,) in results:
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts
    finally:
        db.close()


def get_dominant_emotion(user_id: int, season: str, season_year: int) -> str | None:
    """Most frequent emotion for a season."""
    counts = get_emotion_counts(user_id, season, season_year)
    if not counts:
        return None
    return max(counts, key=counts.get)


def get_average_scores(user_id: int, season: str, season_year: int) -> dict:
    """Average score per emotion across a season (latest predictions only)."""
    db = get_db()
    try:
        results = (
            db.query(EmotionResult.scores_json)
            .join(Checkin, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
                EmotionResult.is_latest == 1,
            )
            .all()
        )

        if not results:
            return {}

        totals = {}
        count = 0
        for (scores_json,) in results:
            if scores_json:
                scores = json.loads(scores_json)
                for emotion, score in scores.items():
                    totals[emotion] = totals.get(emotion, 0) + score
                count += 1

        if count == 0:
            return {}
        return {emotion: round(total / count, 4) for emotion, total in totals.items()}
    finally:
        db.close()


# ── Seasonal summaries ──────────────────────────────────────

def update_seasonal_summary(user_id: int, season: str, season_year: int) -> SeasonalSummary:
    """Recompute and upsert the seasonal summary for a user."""
    db = get_db()
    try:
        avg_scores = get_average_scores(user_id, season, season_year)
        counts = get_emotion_counts(user_id, season, season_year)
        total = sum(counts.values())
        dominant = max(counts, key=counts.get) if counts else None

        avg_sadness = avg_scores.get("sad", 0)
        depression_flag = 1 if avg_sadness > DEPRESSION_THRESHOLD else 0

        summary = (
            db.query(SeasonalSummary)
            .filter(
                SeasonalSummary.user_id == user_id,
                SeasonalSummary.season == season,
                SeasonalSummary.season_year == season_year,
            )
            .first()
        )

        if summary:
            summary.total_checkins = total
            summary.avg_happiness = avg_scores.get("happy", 0)
            summary.avg_sadness = avg_sadness
            summary.dominant_emotion = dominant
            summary.depression_flag = depression_flag
            summary.updated_at = now_iso()
        else:
            summary = SeasonalSummary(
                user_id=user_id,
                season=season,
                season_year=season_year,
                total_checkins=total,
                avg_happiness=avg_scores.get("happy", 0),
                avg_sadness=avg_sadness,
                dominant_emotion=dominant,
                depression_flag=depression_flag,
            )
            db.add(summary)

        db.commit()
        db.refresh(summary)
        return summary
    finally:
        db.close()
