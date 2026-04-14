"""
VibeChecker AI — Database Helper Functions
This is the module the backend team (Zem) imports to interact with the database.
No raw SQL needed — just call these functions.

Usage:
    from db import create_user, create_checkin, get_user_history, ...
"""

import json
from models import get_db, User, Checkin, EmotionResult, SeasonalSummary, now_iso


# ═══════════════════════════════════════════════════════════
# USER OPERATIONS
# ═══════════════════════════════════════════════════════════

def create_user(username: str, email: str, password_hash: str, tz: str = "UTC") -> User:
    """
    Create a new user account.
    NOTE: Hash the password with bcrypt BEFORE calling this function.
    """
    db = get_db()
    try:
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            timezone=tz,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def get_user_by_email(email: str) -> User | None:
    """Look up a user by email (for login)."""
    db = get_db()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def get_user_by_id(user_id: int) -> User | None:
    """Look up a user by ID."""
    db = get_db()
    try:
        return db.query(User).filter(User.user_id == user_id).first()
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# CHECK-IN OPERATIONS
# ═══════════════════════════════════════════════════════════

def create_checkin(user_id: int, image_path: str, captured_at: str,
                   season: str, season_year: int) -> Checkin:
    """
    Record a new daily check-in (one per selfie upload).
    Returns the created Checkin object with its checkin_id.
    """
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


# ═══════════════════════════════════════════════════════════
# EMOTION RESULT OPERATIONS
# ═══════════════════════════════════════════════════════════

def store_emotion_result(checkin_id: int, predicted_emotion: str,
                         confidence: float, scores: dict,
                         model_version: str = "v1.0") -> EmotionResult:
    """
    Store the model's prediction for a check-in.
    `scores` should be a dict like {"happy": 0.7, "sad": 0.1, ...}
    """
    db = get_db()
    try:
        result = EmotionResult(
            checkin_id=checkin_id,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            scores_json=json.dumps(scores),
            model_version=model_version,
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# QUERY OPERATIONS (for dashboard and history)
# ═══════════════════════════════════════════════════════════

def get_user_history(user_id: int, season: str, season_year: int) -> list[dict]:
    """
    Get all check-ins + emotion results for a user in a given season.
    Returns a list of dicts ready for the frontend.
    """
    db = get_db()
    try:
        results = (
            db.query(Checkin, EmotionResult)
            .join(EmotionResult, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
            )
            .order_by(Checkin.captured_at.asc())
            .all()
        )

        return [
            {
                "date": checkin.captured_at,
                "emotion": result.predicted_emotion,
                "confidence": result.confidence,
                "scores": json.loads(result.scores_json) if result.scores_json else {},
                "image_path": checkin.image_path,
            }
            for checkin, result in results
        ]
    finally:
        db.close()


def get_emotion_counts(user_id: int, season: str, season_year: int) -> dict:
    """
    Count occurrences of each emotion for a season.
    Returns: {"happy": 25, "sad": 10, "neutral": 40, ...}
    """
    db = get_db()
    try:
        results = (
            db.query(EmotionResult.predicted_emotion)
            .join(Checkin, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
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
    """Get the most frequent emotion for a user in a given season."""
    counts = get_emotion_counts(user_id, season, season_year)
    if not counts:
        return None
    return max(counts, key=counts.get)


def get_average_scores(user_id: int, season: str, season_year: int) -> dict:
    """
    Calculate average score for each emotion across a season.
    Returns: {"happy": 0.23, "sad": 0.31, ...}
    """
    db = get_db()
    try:
        results = (
            db.query(EmotionResult.scores_json)
            .join(Checkin, Checkin.checkin_id == EmotionResult.checkin_id)
            .filter(
                Checkin.user_id == user_id,
                Checkin.season == season,
                Checkin.season_year == season_year,
            )
            .all()
        )

        if not results:
            return {}

        # Accumulate scores
        totals = {}
        count = 0
        for (scores_json,) in results:
            if scores_json:
                scores = json.loads(scores_json)
                for emotion, score in scores.items():
                    totals[emotion] = totals.get(emotion, 0) + score
                count += 1

        # Average them
        if count == 0:
            return {}
        return {emotion: round(total / count, 4) for emotion, total in totals.items()}
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# SEASONAL SUMMARY OPERATIONS
# ═══════════════════════════════════════════════════════════

def update_seasonal_summary(user_id: int, season: str, season_year: int) -> SeasonalSummary:
    """
    Recalculate and store the seasonal summary for a user.
    Call this after new check-ins are added, or on a schedule.
    """
    db = get_db()
    try:
        avg_scores = get_average_scores(user_id, season, season_year)
        counts = get_emotion_counts(user_id, season, season_year)
        total = sum(counts.values())
        dominant = max(counts, key=counts.get) if counts else None

        # Flag depression if average sadness is above threshold
        avg_sadness = avg_scores.get("sad", 0)
        depression_flag = 1 if avg_sadness > 0.3 else 0  # Threshold is adjustable

        # Update existing or create new
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
