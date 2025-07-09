import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    pending_email = Column(String, nullable=True)
    email_change_token = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    pwd_reset_requested_at = Column(DateTime, nullable=True)

    chargers = relationship(
        "Charger", back_populates="owner", cascade="all, delete-orphan"
    )


class Charger(Base):
    __tablename__ = "chargers"

    id = Column(Integer, primary_key=True, index=True)
    charger_id = Column(String, nullable=False)
    description = Column(String, nullable=True)
    cost_kwh = Column(Float, nullable=False)
    registered_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    owner = relationship("User", back_populates="chargers")
    sessions = relationship(
        "ChargingSession", back_populates="charger", cascade="all, delete-orphan"
    )
    events = relationship(
        "ChargingEvent", back_populates="charger", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("charger_id", "owner_id", name="uix_charger_owner"),
    )


class ChargingStatus(Base):
    __tablename__ = "charging_statuses"

    id = Column(Integer, primary_key=True, index=True)
    charger_id = Column(Integer, ForeignKey("chargers.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    cost = Column(Float, nullable=False)

    seconds = Column(Integer, nullable=False)
    energy = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    phase_count = Column(Integer, nullable=False)

    charger = relationship("Charger", backref="statuses")


class ChargingEvent(Base):
    __tablename__ = "charging_events"

    id = Column(Integer, primary_key=True, index=True)
    charger_id = Column(Integer, ForeignKey("chargers.id"), nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)

    charger = relationship("Charger", back_populates="events")


class ChargingSession(Base):
    __tablename__ = "charging_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    charger_id = Column(Integer, ForeignKey("chargers.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    energy_charged_kwh = Column(Float, nullable=False)
    cost =  Column(Float, nullable=False)
    tag = Column(String(30), nullable=True)

    charger = relationship("Charger", back_populates="sessions")
