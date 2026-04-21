"""SQLAlchemy models for VibeChecker AI."""

import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Text, Float, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

DATABASE_URL = "sqlite:///vibechecker.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Return a new DB session. Caller must close it."""
    return SessionLocal()


def now_iso():
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(Text, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)  # bcrypt hash
    created_at = Column(Text, default=now_iso)
    timezone = Column(Text, default="UTC")

    checkins = relationship("Checkin", back_populates="user")
    seasonal_summaries = relationship("SeasonalSummary", back_populates="user")

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "timezone": self.timezone,
        }  # password_hash excluded on purpose

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


class Checkin(Base):
    __tablename__ = "checkins"

    checkin_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    image_path = Column(Text, nullable=False)
    captured_at = Column(Text, nullable=False)
    season = Column(Text, nullable=False)
    season_year = Column(Integer, nullable=False)
    created_at = Column(Text, default=now_iso)

    user = relationship("User", back_populates="checkins")

    # All predictions ever made for this check-in (newest first).
    emotion_results = relationship(
        "EmotionResult",
        back_populates="checkin",
        order_by="EmotionResult.processed_at.desc()",
    )

    # Convenience: only the current (is_latest=1) prediction, or None.
    latest_result = relationship(
        "EmotionResult",
        primaryjoin="and_(Checkin.checkin_id==EmotionResult.checkin_id, EmotionResult.is_latest==1)",
        uselist=False,
        viewonly=True,
    )

    def to_dict(self):
        return {
            "checkin_id": self.checkin_id,
            "user_id": self.user_id,
            "image_path": self.image_path,
            "captured_at": self.captured_at,
            "season": self.season,
            "season_year": self.season_year,
            "created_at": self.created_at,
        }

    def __repr__(self):
        return f"<Checkin(checkin_id={self.checkin_id}, user_id={self.user_id}, season='{self.season}')>"


class EmotionResult(Base):
    __tablename__ = "emotion_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    checkin_id = Column(Integer, ForeignKey("checkins.checkin_id"), nullable=False)
    predicted_emotion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    scores_json = Column(Text)  # JSON: {"happy": 0.7, "sad": 0.1, ...}
    model_version = Column(Text)
    is_latest = Column(Integer, nullable=False, default=1)  # 1 = current, 0 = historical
    processed_at = Column(Text, default=now_iso)

    checkin = relationship("Checkin", back_populates="emotion_results")

    def to_dict(self):
        return {
            "result_id": self.result_id,
            "checkin_id": self.checkin_id,
            "predicted_emotion": self.predicted_emotion,
            "confidence": self.confidence,
            "scores": json.loads(self.scores_json) if self.scores_json else {},
            "model_version": self.model_version,
            "is_latest": self.is_latest,
            "processed_at": self.processed_at,
        }

    def __repr__(self):
        return f"<EmotionResult(result_id={self.result_id}, emotion='{self.predicted_emotion}', latest={self.is_latest})>"


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
    depression_flag = Column(Integer, default=0)  # 1 = flagged
    updated_at = Column(Text, default=now_iso)

    user = relationship("User", back_populates="seasonal_summaries")

    def to_dict(self):
        return {
            "summary_id": self.summary_id,
            "user_id": self.user_id,
            "season": self.season,
            "season_year": self.season_year,
            "total_checkins": self.total_checkins,
            "avg_happiness": self.avg_happiness,
            "avg_sadness": self.avg_sadness,
            "dominant_emotion": self.dominant_emotion,
            "depression_flag": self.depression_flag,
            "updated_at": self.updated_at,
        }

    def __repr__(self):
        return f"<SeasonalSummary(user_id={self.user_id}, season='{self.season} {self.season_year}')>"


Index("idx_checkins_user_season", Checkin.user_id, Checkin.season, Checkin.season_year)
Index("idx_checkins_captured", Checkin.captured_at)
Index("idx_emotion_results_checkin_latest", EmotionResult.checkin_id, EmotionResult.is_latest)
