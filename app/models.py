"""Relational schema (Phase 1).

users ─< bookings >─ flights >─ airports (origin / dest)
                │        └──── aircraft (seat map layout)
                ├─< passengers   UNIQUE(flight_id, seat_no)   <- final double-booking guard
                ├─< seat_locks   UNIQUE(flight_id, seat_no)   <- 10-minute holds
                └─< payments     one row per group member (split checkout)
"""
from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Index, Integer,
                        String, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    pw_hash: Mapped[str] = mapped_column(String(200))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)


class Airport(Base):
    __tablename__ = "airports"
    code: Mapped[str] = mapped_column(String(3), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    city: Mapped[str] = mapped_column(String(80))
    size: Mapped[int] = mapped_column(Integer, default=2)  # 1 small, 2 medium, 3 hub


class Aircraft(Base):
    __tablename__ = "aircraft"
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(60))
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)  # seats abreast (4 or 6)
    business_rows: Mapped[int] = mapped_column(Integer, default=2)

    @property
    def capacity(self) -> int:
        return self.rows * self.cols


class Flight(Base):
    __tablename__ = "flights"
    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(8), index=True)
    origin: Mapped[str] = mapped_column(ForeignKey("airports.code"))
    dest: Mapped[str] = mapped_column(ForeignKey("airports.code"))
    depart: Mapped[datetime] = mapped_column(DateTime, index=True)
    arrive: Mapped[datetime] = mapped_column(DateTime)
    base_price: Mapped[int] = mapped_column(Integer)  # economy base fare, PHP
    aircraft_id: Mapped[int] = mapped_column(ForeignKey("aircraft.id"))

    aircraft: Mapped[Aircraft] = relationship(lazy="joined")
    origin_airport: Mapped[Airport] = relationship(foreign_keys=[origin], lazy="joined")
    dest_airport: Mapped[Airport] = relationship(foreign_keys=[dest], lazy="joined")

    __table_args__ = (Index("ix_flight_route", "origin", "dest", "depart"),)

    @property
    def minutes(self) -> int:
        return int((self.arrive - self.depart).total_seconds() // 60)


class Booking(Base):
    __tablename__ = "bookings"
    id: Mapped[int] = mapped_column(primary_key=True)
    ref: Mapped[str] = mapped_column(String(8), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id"), index=True)
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending|confirmed|expired
    total_amount: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    flight: Mapped[Flight] = relationship(lazy="joined")
    user: Mapped[User] = relationship()
    passengers: Mapped[list["Passenger"]] = relationship(back_populates="booking", order_by="Passenger.seat_no")
    payments: Mapped[list["Payment"]] = relationship(back_populates="booking", order_by="Payment.id")


class Passenger(Base):
    """A confirmed seat. The UNIQUE constraint is the last line of defence
    against double-booking, even if application logic were bypassed."""
    __tablename__ = "passengers"
    id: Mapped[int] = mapped_column(primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"), index=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    seat_no: Mapped[str] = mapped_column(String(4))
    price: Mapped[int] = mapped_column(Integer)

    booking: Mapped[Booking] = relationship(back_populates="passengers")
    __table_args__ = (UniqueConstraint("flight_id", "seat_no", name="uq_passenger_seat"),)


class SeatLock(Base):
    """Temporary hold. INSERT either wins the unique constraint or fails
    atomically - this is what makes seat selection race-condition free."""
    __tablename__ = "seat_locks"
    id: Mapped[int] = mapped_column(primary_key=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id"), index=True)
    seat_no: Mapped[str] = mapped_column(String(4))
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"), index=True)
    passenger_name: Mapped[str] = mapped_column(String(120), default="")
    price: Mapped[int] = mapped_column(Integer)  # price is frozen at hold time
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (UniqueConstraint("flight_id", "seat_no", name="uq_lock_seat"),)


class Payment(Base):
    """One row per group member. Each has a unique checkout link (token)."""
    __tablename__ = "payments"
    id: Mapped[int] = mapped_column(primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"), index=True)
    token: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(60), default="Payer")
    amount: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending|paid|refunded
    method: Mapped[str] = mapped_column(String(12), default="")
    last4: Mapped[str] = mapped_column(String(4), default="")  # never store full card numbers
    provider_ref: Mapped[str] = mapped_column(String(40), default="")
    payer_name: Mapped[str] = mapped_column(String(120), default="")
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    booking: Mapped[Booking] = relationship(back_populates="payments", lazy="joined")
