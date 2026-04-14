"""
VibeChecker AI — SQLAlchemy Models
Defines all database tables as Python classes.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, Text, Float, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ── Base setup ──────────────────────────────────────────────
Base = declarative_base()

# ── Database connection ─────────────────────────────────────
# SQLite stores everything in this single file
DATABASE_URL = "sqlite:///vibechecker.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Get a database session. Use in a with-block or call .close() when done."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def now_iso():
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Table: users ────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(Text, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)  # Use bcrypt to hash before storing
    created_at = Column(Text, default=now_iso)
    timezone = Column(Text, default="UTC")

    # Relationships — lets you do user.checkins to get all their check-ins
    checkins = relationship("Checkin", back_populates="user")
    seasonal_summaries = relationship("SeasonalSummary", back_populates="user")

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


# ── Table: checkins ─────────────────────────────────────────
class Checkin(Base):
    __tablename__ = "checkins"

    checkin_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    image_path = Column(Text, nullable=False)
    captured_at = Column(Text, nullable=False)  # ISO 8601 timestamp
    season = Column(Text, nullable=False)        # "winter", "spring", "summer", "fall"
    season_year = Column(Integer, nullable=False)
    created_at = Column(Text, default=now_iso)

    # Relationships
    user = relationship("User", back_populates="checkins")
    emotion_result = relationship("EmotionResult", back_populates="checkin", uselist=False)

    def __repr__(self):
        return f"<Checkin(checkin_id={self.checkin_id}, user_id={self.user_id}, season='{self.season}')>"


# ── Table: emotion_results ──────────────────────────────────
class EmotionResult(Base):
    __tablename__ = "emotion_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    checkin_id = Column(Integer, ForeignKey("checkins.checkin_id"), nullable=False)
    predicted_emotion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    scores_json = Column(Text)          # JSON string: {"happy": 0.7, "sad": 0.1, ...}
    model_version = Column(Text)
    processed_at = Column(Text, default=now_iso)

    # Relationship
    checkin = relationship("Checkin", back_populates="emotion_result")

    def __repr__(self):
        return f"<EmotionResult(result_id={self.result_id}, emotion='{self.predicted_emotion}')>"


# ── Table: seasonal_summaries ───────────────────────────────
class SeasonalSummary(Base):
    __tablename__ = "seasonal_summaries"

    summary_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    season = Column(Text, nullable=False)
    season_year = Column(Integer, nullable=False)
    total_checkins = Column(Integer, default=0)
    avg_happiness = Column(Float, default=0.0)
    avg_sadness = Column(Float, default=0.0)
    dominant_emotion = Column(Text)
    depression_flag = Column(Integer, default=0)  # 1 = flagged, 0 = normal
    updated_at = Column(Text, default=now_iso)

    # Relationship
    user = relationship("User", back_populates="seasonal_summaries")

    def __repr__(self):
        return f"<SeasonalSummary(user_id={self.user_id}, season='{self.season} {self.season_year}')>"


# ── Indexes for fast queries ────────────────────────────────
Index("idx_checkins_user_season", Checkin.user_id, Checkin.season, Checkin.season_year)
Index("idx_checkins_captured", Checkin.captured_at)
Index("idx_emotion_results_checkin", EmotionResult.checkin_id)
