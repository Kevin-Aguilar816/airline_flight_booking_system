"""Business logic: pricing, seat maps, concurrency-safe seat holds,
split-payment checkout and booking finalisation (Phase 2 + 3)."""
import os
import secrets
import string
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from .db import utcnow
from .models import Booking, Flight, Passenger, Payment, SeatLock
from .payments import gateway

HOLD_MINUTES = int(os.getenv("HOLD_MINUTES", "10"))
MAX_SEATS = 9
LETTERS = "ABCDEFGH"


class SeatUnavailable(Exception):
    pass


class CheckoutError(Exception):
    pass


# ----------------------------------------------------------------- pricing --
def seat_ids(aircraft) -> list[str]:
    return [f"{r}{LETTERS[c]}" for r in range(1, aircraft.rows + 1) for c in range(aircraft.cols)]


def seat_multiplier(aircraft, row: int) -> float:
    if row <= aircraft.business_rows:
        return 2.2
    if row <= aircraft.business_rows + 3:
        return 1.25  # extra-legroom economy
    return 1.0


def seat_class(aircraft, row: int) -> str:
    if row <= aircraft.business_rows:
        return "business"
    if row <= aircraft.business_rows + 3:
        return "legroom"
    return "economy"


def loads_for(db, flights) -> dict[int, float]:
    """Load factor (0..1) per flight from confirmed passengers - one query."""
    ids = [f.id for f in flights]
    counts = {}
    if ids:
        counts = dict(db.execute(select(Passenger.flight_id, func.count())
                                 .where(Passenger.flight_id.in_(ids))
                                 .group_by(Passenger.flight_id)).all())
    return {f.id: counts.get(f.id, 0) / f.aircraft.capacity for f in flights}


def dynamic_fare(flight: Flight, load: float, now: datetime | None = None) -> int:
    """Economy fare = base fare x demand multiplier (load factor + urgency)."""
    now = now or datetime.now()
    days = max((flight.depart - now).total_seconds() / 86400, 0)
    urgency = 0.35 if days < 3 else 0.20 if days < 7 else 0.10 if days < 14 else -0.05 if days > 45 else 0.0
    mult = 1 + 0.6 * load ** 2 + urgency
    return int(round(flight.base_price * mult / 10.0) * 10)


# --------------------------------------------------------------- seat map ---
def pending_booking(db, user, flight_id) -> Booking | None:
    if not user:
        return None
    return db.scalar(select(Booking).where(Booking.user_id == user.id, Booking.flight_id == flight_id,
                                           Booking.status == "pending"))


def build_seatmap(db, flight: Flight, user) -> dict:
    ac = flight.aircraft
    now = utcnow()
    taken = {s for (s,) in db.execute(select(Passenger.seat_no).where(Passenger.flight_id == flight.id))}
    locks = db.scalars(select(SeatLock).where(SeatLock.flight_id == flight.id, SeatLock.expires_at > now)).all()
    booking = pending_booking(db, user, flight.id)
    mine = {l.seat_no: l for l in locks if booking and l.booking_id == booking.id}
    others = {l.seat_no for l in locks if l.seat_no not in mine}
    load = len(taken) / ac.capacity
    fare = dynamic_fare(flight, load)

    rows = []
    for r in range(1, ac.rows + 1):
        seats = []
        for c in range(ac.cols):
            sid = f"{r}{LETTERS[c]}"
            price = mine[sid].price if sid in mine else int(round(fare * seat_multiplier(ac, r) / 10) * 10)
            state = "taken" if sid in taken else "mine" if sid in mine else "locked" if sid in others else "free"
            seats.append({"id": sid, "state": state, "price": price, "cls": seat_class(ac, r)})
        rows.append({"row": r, "seats": seats})
    return {
        "rows": rows, "aisle_after": ac.cols // 2, "letters": LETTERS[:ac.cols],
        "mine": sorted(mine.values(), key=lambda l: (int(l.seat_no[:-1]), l.seat_no[-1])),
        "total": sum(l.price for l in mine.values()),
        "expires": min((l.expires_at for l in mine.values()), default=None),
        "booking": booking, "fare": fare, "load": load,
        "free": ac.capacity - len(taken) - len(others) - len(mine),
        "taken": len(taken), "held": len(others),
    }


# ------------------------------------------------- concurrency-safe holds ---
def _ref() -> str:
    return "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def get_or_create_pending(db, user, flight_id) -> Booking:
    b = pending_booking(db, user, flight_id)
    if b:
        return b
    b = Booking(ref=_ref(), user_id=user.id, flight_id=flight_id, status="pending")
    db.add(b)
    db.commit()
    return b


def _drop_unpaid_payments(db, booking: Booking):
    """Seats changed -> any generated split links are stale."""
    if any(p.status == "paid" for p in booking.payments):
        raise SeatUnavailable("Payment already started for this booking - seats can no longer be changed.")
    db.execute(delete(Payment).where(Payment.booking_id == booking.id))


def hold_seat(db, flight: Flight, seat: str, user, minutes: int | None = None) -> SeatLock:
    """Atomically hold a seat for HOLD_MINUTES.

    Safety argument: the INSERT into seat_locks is protected by
    UNIQUE(flight_id, seat_no). If two users click the same seat at the same
    instant, the database lets exactly ONE insert succeed; the other gets an
    IntegrityError, which we translate into SeatUnavailable.
    Expired locks are purged first so stale holds never block a seat.
    """
    if seat not in seat_ids(flight.aircraft):
        raise SeatUnavailable("That seat does not exist.")
    if flight.depart < datetime.now():
        raise SeatUnavailable("This flight has already departed.")
    booking = get_or_create_pending(db, user, flight.id)
    _drop_unpaid_payments(db, booking)
    db.commit()

    now = utcnow()
    db.execute(delete(SeatLock).where(SeatLock.flight_id == flight.id, SeatLock.expires_at < now))
    if db.scalar(select(Passenger.id).where(Passenger.flight_id == flight.id, Passenger.seat_no == seat)):
        db.rollback()
        raise SeatUnavailable("Seat already booked.")
    mine = db.scalar(select(func.count()).select_from(SeatLock).where(SeatLock.booking_id == booking.id))
    if mine >= MAX_SEATS:
        db.rollback()
        raise SeatUnavailable(f"You can hold at most {MAX_SEATS} seats per booking.")

    load = loads_for(db, [flight])[flight.id]
    price = int(round(dynamic_fare(flight, load) * seat_multiplier(flight.aircraft, int(seat[:-1])) / 10) * 10)
    lock = SeatLock(flight_id=flight.id, seat_no=seat, booking_id=booking.id, price=price,
                    expires_at=now + timedelta(minutes=minutes or HOLD_MINUTES))
    db.add(lock)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise SeatUnavailable("Someone else just took that seat.")
    return lock


def release_seat(db, flight: Flight, seat: str, user):
    booking = pending_booking(db, user, flight.id)
    if not booking:
        return
    _drop_unpaid_payments(db, booking)
    db.execute(delete(SeatLock).where(SeatLock.booking_id == booking.id, SeatLock.seat_no == seat))
    db.commit()


def active_locks(db, booking: Booking) -> list[SeatLock]:
    return db.scalars(select(SeatLock).where(SeatLock.booking_id == booking.id, SeatLock.expires_at > utcnow())
                      .order_by(SeatLock.id)).all()


# ---------------------------------------------------- split-pay checkout ----
def prepare_checkout(db, booking: Booking, names: dict[str, str], payers: int) -> list[Payment]:
    locks = active_locks(db, booking)
    if not locks:
        raise CheckoutError("Your seat hold expired. Please select seats again.")
    if any(p.status == "paid" for p in booking.payments):
        raise CheckoutError("Payment has already started.")
    payers = max(1, min(payers, len(locks)))
    for l in locks:
        n = (names.get(l.seat_no) or "").strip()
        if not n:
            raise CheckoutError(f"Enter a passenger name for seat {l.seat_no}.")
        l.passenger_name = n[:120]
    total = sum(l.price for l in locks)
    booking.total_amount = total
    db.execute(delete(Payment).where(Payment.booking_id == booking.id))
    share, extra = divmod(total, payers)
    pays = []
    for i in range(payers):
        p = Payment(booking_id=booking.id, token=secrets.token_urlsafe(8), label=f"Payer {i + 1}",
                    amount=share + (extra if i == 0 else 0))
        db.add(p)
        pays.append(p)
    db.commit()
    return pays


def pay(db, payment: Payment, method: str, card_number: str, payer_name: str) -> str:
    """Charge one share. Returns the booking status afterwards."""
    booking = payment.booking
    if payment.status != "pending":
        raise CheckoutError("This share has already been paid.")
    if booking.status != "pending":
        raise CheckoutError(f"This booking is {booking.status}.")
    locks = active_locks(db, booking)
    if not locks or sum(l.price for l in locks) != booking.total_amount:
        raise CheckoutError("The seat hold expired before payment. Nothing was charged.")
    result = gateway.charge(method, card_number, payment.amount)
    if not result.ok:
        raise CheckoutError(f"Payment failed: {result.message}")
    payment.status, payment.method = "paid", method
    payment.last4, payment.provider_ref = result.last4, result.ref
    payment.payer_name = (payer_name or payment.label).strip()[:120]
    payment.paid_at = utcnow()
    db.commit()
    return finalize_if_paid(db, booking.id)


def _expire_and_refund(db, booking_id: int) -> str:
    b = db.get(Booking, booking_id)
    for p in b.payments:
        if p.status == "paid":
            p.status = "refunded"
    b.status = "expired"
    db.execute(delete(SeatLock).where(SeatLock.booking_id == b.id))
    db.commit()
    return "expired"


def finalize_if_paid(db, booking_id: int) -> str:
    """When the LAST group member pays, convert holds into passengers in one
    transaction. FOR UPDATE serialises concurrent finalisers on PostgreSQL."""
    b = db.execute(select(Booking).where(Booking.id == booking_id).with_for_update(of=Booking)).scalar_one()
    if b.status != "pending":
        db.rollback()
        return b.status
    if not b.payments or any(p.status != "paid" for p in b.payments):
        db.rollback()
        return "pending"
    now = utcnow()
    locks = db.scalars(select(SeatLock).where(SeatLock.booking_id == b.id)).all()
    if not locks or any(l.expires_at < now for l in locks) or sum(l.price for l in locks) != b.total_amount:
        db.rollback()
        return _expire_and_refund(db, booking_id)
    for l in locks:
        db.add(Passenger(booking_id=b.id, flight_id=b.flight_id, name=l.passenger_name,
                         seat_no=l.seat_no, price=l.price))
        db.delete(l)
    b.status = "confirmed"
    try:
        db.commit()
    except IntegrityError:  # seat somehow already sold -> refund everyone
        db.rollback()
        return _expire_and_refund(db, booking_id)
    from . import indexes          # local import: indexes only depends on models and dsa
    indexes.on_confirmed(db, b)
    return "confirmed"
