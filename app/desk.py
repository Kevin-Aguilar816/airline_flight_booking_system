"""Booking desk: priority requests, waitlists, undo, manifests.

How the pieces fit together
---------------------------
  traveler request --> MaxHeap (tier, then arrival) --> agent "Process next"
        seats free?  yes -> seats HELD for the traveler (they pay on the normal checkout)
                     no  -> flight WAITLIST (Deque, strict first-in first-out)
  cancellation or "Undo last booking" (Stack, LIFO)
        --> seats released --> waitlist promoted from the FRONT of the deque
  flight manifest = SinglyLinkedList of bookings; a cancelled booking is detached
"""
import secrets
from datetime import datetime

from sqlalchemy import select

from . import indexes, services
from .db import utcnow
from .dsa import Deque, MaxHeap, SinglyLinkedList
from .models import Booking, BookingRequest, Flight, Passenger, Payment, SeatLock

TIERS = {"VIP": 3, "Business": 2, "Economy": 1}
# which cabins each tier may use, best first (a tier never takes a better cabin than its own)
TIER_CABINS = {"VIP": ["business", "legroom", "economy"], "Business": ["legroom", "economy"], "Economy": ["economy"]}
OFFER_MINUTES = 30      # how long desk-allocated seats are held for the traveler
MAX_REQUEST_SEATS = 9


class DeskError(Exception):
    pass


# ------------------------------------------------------------ seat picking ---
def free_seats_by_cabin(db, flight: Flight) -> dict[str, list[str]]:
    ac, now = flight.aircraft, utcnow()
    taken = {s for (s,) in db.execute(select(Passenger.seat_no).where(Passenger.flight_id == flight.id))}
    held = {s for (s,) in db.execute(select(SeatLock.seat_no).where(SeatLock.flight_id == flight.id,
                                                                      SeatLock.expires_at > now))}
    out: dict[str, list[str]] = {"business": [], "legroom": [], "economy": []}
    for seat in services.seat_ids(ac):
        if seat not in taken and seat not in held:
            out[services.seat_class(ac, int(seat[:-1]))].append(seat)
    return out


def is_full(db, flight: Flight) -> bool:
    return not any(free_seats_by_cabin(db, flight).values())


def _adjacent_block(seats: list[str], n: int) -> list[str] | None:
    """n seats side by side in one row (consecutive letters), if any."""
    rows: dict[int, list[str]] = {}
    for s in seats:
        rows.setdefault(int(s[:-1]), []).append(s)
    for r in sorted(rows):
        row = sorted(rows[r], key=lambda s: s[-1])
        for i in range(len(row) - n + 1):
            window = row[i:i + n]
            if ord(window[-1][-1]) - ord(window[0][-1]) == n - 1:
                return window
    return None


def pick_seats(db, flight: Flight, tier: str, n: int) -> list[str] | None:
    free = free_seats_by_cabin(db, flight)
    order = TIER_CABINS[tier]
    for cabin in order:                                   # best: together, in the tier's own cabin
        block = _adjacent_block(free[cabin], n)
        if block:
            return block
    pool = [s for cabin in order for s in free[cabin]]    # otherwise: any seats the tier may use
    return pool[:n] if len(pool) >= n else None


def allocate(db, req: BookingRequest) -> list[str] | None:
    """Hold seats for the request's user. Returns the seats, or None if they do not fit."""
    seats = pick_seats(db, req.flight, req.tier, req.seats)
    if not seats:
        return None
    got = []
    try:
        for s in seats:
            services.hold_seat(db, req.flight, s, req.user, minutes=OFFER_MINUTES)
            got.append(s)
    except services.SeatUnavailable:                      # lost a race: undo and report "no seats"
        for s in got:
            services.release_seat(db, req.flight, s, req.user)
        return None
    booking = services.pending_booking(db, req.user, req.flight_id)
    req.status, req.booking_id = "allocated", booking.id
    req.note = "Seats " + ", ".join(sorted(got)) + f" held {OFFER_MINUTES} min: complete checkout"
    db.commit()
    return got


# ---------------------------------------------------------------- requests ---
def _check(tier: str, seats: int):
    if tier not in TIERS:
        raise DeskError("Choose a ticket tier: VIP, Business or Economy.")
    if not 1 <= seats <= MAX_REQUEST_SEATS:
        raise DeskError(f"Request between 1 and {MAX_REQUEST_SEATS} seats.")


def submit_request(db, user, flight: Flight, tier: str, seats: int) -> BookingRequest:
    _check(tier, seats)
    if flight.depart < datetime.now():
        raise DeskError("This flight has already departed.")
    req = BookingRequest(user_id=user.id, flight_id=flight.id, tier=tier, seats=seats, status="queued",
                         note="Waiting for the booking desk")
    db.add(req)
    db.commit()
    return req


def join_waitlist(db, user, flight: Flight, tier: str, seats: int) -> BookingRequest:
    _check(tier, seats)
    if flight.depart < datetime.now():
        raise DeskError("This flight has already departed.")
    if not is_full(db, flight):
        raise DeskError("Seats are still available on this flight: pick them on the seat map "
                        "or send a priority request instead.")
    req = BookingRequest(user_id=user.id, flight_id=flight.id, tier=tier, seats=seats, status="waitlisted",
                         waitlisted_at=utcnow(), note="Flight full: in the waitlist line")
    db.add(req)
    db.commit()
    return req


def cancel_request(db, req: BookingRequest, user) -> None:
    if req.user_id != user.id and not user.is_admin:
        raise DeskError("Not your request.")
    if req.status not in ("queued", "waitlisted"):
        raise DeskError("Only waiting requests can be withdrawn.")
    req.status, req.note = "cancelled", "Withdrawn by the traveler"
    db.commit()


def queued_heap(db) -> MaxHeap:
    """Priority = (tier rank, -arrival id): VIP before Business before Economy,
    and within a tier the earlier request wins."""
    heap = MaxHeap()
    for r in db.scalars(select(BookingRequest).where(BookingRequest.status == "queued")).unique():
        heap.push((TIERS[r.tier], -r.id), r)
    return heap


def process_next(db) -> dict | None:
    heap = queued_heap(db)
    if not len(heap):
        return None
    req = heap.pop()                                      # highest priority
    seats = allocate(db, req)
    if seats:
        return {"request": req, "action": "allocated", "seats": seats}
    req.status, req.waitlisted_at = "waitlisted", utcnow()
    req.note = "No seats for this tier: moved to the waitlist"
    db.commit()
    return {"request": req, "action": "waitlisted", "seats": []}


# ---------------------------------------------------------------- waitlist ---
def expire_offers(db) -> None:
    """Lazy housekeeping: close allocated requests whose hold ran out or that were paid."""
    for req in db.scalars(select(BookingRequest).where(BookingRequest.status == "allocated")).unique():
        b = db.get(Booking, req.booking_id) if req.booking_id else None
        if b is None:
            continue
        if b.status == "confirmed":
            req.status, req.note = "booked", "Booking confirmed"
        elif b.status == "pending" and not services.active_locks(db, b):
            req.status, req.note, b.status = "expired", "Hold expired before payment", "expired"
    db.commit()


def waitlist_deque(db, flight_id: int) -> Deque:
    dq = Deque()
    rows = db.scalars(select(BookingRequest).where(BookingRequest.flight_id == flight_id,
                                                   BookingRequest.status == "waitlisted")
                      .order_by(BookingRequest.waitlisted_at, BookingRequest.id)).unique().all()
    for r in rows:
        dq.push_back(r)                                   # join at the back
    return dq


def waitlist_position(db, req: BookingRequest) -> int | None:
    for i, r in enumerate(waitlist_deque(db, req.flight_id), start=1):
        if r.id == req.id:
            return i
    return None


def process_waitlist(db, flight_id: int) -> list[BookingRequest]:
    """Offer freed seats to the FRONT of the line. Strict first-in, first-out:
    if the person at the front does not fit yet, nobody behind them jumps ahead."""
    expire_offers(db)
    dq = waitlist_deque(db, flight_id)
    promoted = []
    while len(dq):
        if allocate(db, dq.peek_front()):
            promoted.append(dq.pop_front())
        else:
            break
    return promoted


def all_waitlists(db) -> list[dict]:
    flight_ids = sorted({fid for (fid,) in db.execute(select(BookingRequest.flight_id)
                                                      .where(BookingRequest.status == "waitlisted"))})
    return [{"flight": db.get(Flight, fid), "line": waitlist_deque(db, fid).to_list()} for fid in flight_ids]


# ------------------------------------------------------- cancel / undo ------
def cancel_booking(db, booking_id: int) -> dict:
    b = db.get(Booking, booking_id)
    if not b or b.status != "confirmed":
        raise DeskError("That booking is not confirmed, so there is nothing to cancel.")
    with indexes.lock:
        seats = [p.seat_no for p in b.passengers]
        keys = [indexes.ticket_key(b.ref, s) for s in seats]
        for p in list(b.passengers):
            db.delete(p)                                  # releases the seats
        for pay in b.payments:
            if pay.status == "paid":
                pay.status = "refunded"                   # sandbox refund
        b.status = "cancelled"
        db.commit()
        indexes.forget_passengers(keys)
    promoted = process_waitlist(db, b.flight_id)
    return {"booking": b, "seats": seats, "promoted": promoted}


def undo_last_booking(db) -> dict:
    """Pop the stack until a still-confirmed booking is found (entries cancelled
    elsewhere are skipped), cancel it, and let the waitlist take the seats."""
    with indexes.lock:
        while len(indexes.undo_stack):
            bid = indexes.undo_stack.pop()
            b = db.get(Booking, bid)
            if b and b.status == "confirmed":
                return cancel_booking(db, bid)
    raise DeskError("Nothing to undo: no recent confirmed bookings.")


# ---------------------------------------------------------------- manifest ---
def manifest(db, flight_id: int) -> SinglyLinkedList:
    chain = SinglyLinkedList()
    for b in db.scalars(select(Booking).where(Booking.flight_id == flight_id, Booking.status == "confirmed")
                        .order_by(Booking.id)).unique():
        chain.append({"id": b.id, "ref": b.ref, "name": b.user.name,
                      "seats": sorted(p.seat_no for p in b.passengers), "total": b.total_amount})
    return chain


def cancel_from_manifest(db, flight_id: int, booking_id: int) -> dict:
    chain = manifest(db, flight_id)
    detached = chain.detach(lambda node: node["id"] == booking_id)    # unlink from the chain
    if detached is None:
        raise DeskError("That booking is not on this flight's manifest.")
    result = cancel_booking(db, booking_id)
    result["detached"] = detached
    result["chain_after"] = chain.to_list()
    return result


# -------------------------------------------------------------- demo helper --
def fill_flight(db, flight: Flight, admin) -> int:
    """Demo tool: sell every free seat in one confirmed booking so the flight is
    full and the waitlist can be shown. Returns the number of seats sold."""
    free = free_seats_by_cabin(db, flight)
    seats = [s for cabin in ("business", "legroom", "economy") for s in free[cabin]]
    if not seats:
        raise DeskError("This flight is already full.")
    load = services.loads_for(db, [flight])[flight.id]
    fare = services.dynamic_fare(flight, load)
    ac = flight.aircraft
    prices = {s: int(round(fare * services.seat_multiplier(ac, int(s[:-1])) / 10) * 10) for s in seats}
    b = Booking(ref=services._ref(), user_id=admin.id, flight_id=flight.id, status="confirmed",
                total_amount=sum(prices.values()))
    db.add(b)
    db.flush()
    for i, s in enumerate(seats, start=1):
        db.add(Passenger(booking_id=b.id, flight_id=flight.id, name=f"Demo Passenger {i}", seat_no=s, price=prices[s]))
    db.add(Payment(booking_id=b.id, token=secrets.token_urlsafe(8), label="Demo fill", amount=b.total_amount,
                   status="paid", method="demo", provider_ref="demo", payer_name="Demo", paid_at=utcnow()))
    db.commit()
    indexes.on_confirmed(db, b)
    return len(seats)
