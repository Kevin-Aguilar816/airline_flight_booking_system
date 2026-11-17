"""Proves the double-booking protection: many users hammer the SAME seat at once
and exactly one wins. Run:  pytest -q"""
import os
import tempfile
import threading
from datetime import datetime, timedelta

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MODEL_PATH"] = f"{_tmp}/m.joblib"

import pytest  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app import routing, services  # noqa: E402
from app.db import Base, SessionLocal, engine, utcnow  # noqa: E402
from app.models import Flight, Passenger, SeatLock, User  # noqa: E402
from app.seed import hash_pw, seed  # noqa: E402


@pytest.fixture(scope="module")
def world():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db, history_days_back=1, history_days_fwd=1, days_ahead=3)
        users = [User(email=f"u{i}@t.io", name=f"U{i}", pw_hash=hash_pw("x")) for i in range(25)]
        db.add_all(users)
        db.commit()
        flight = db.scalars(select(Flight).where(Flight.depart > datetime.now() + timedelta(days=2))).unique().first()
        return [u.id for u in users], flight.id


def test_only_one_user_wins_the_same_seat(world):
    user_ids, fid = world
    results, barrier = [], threading.Barrier(len(user_ids))

    def worker(uid):
        barrier.wait()  # release all threads at the same instant (before taking a pooled connection)
        with SessionLocal() as db:
            u, f = db.get(User, uid), db.get(Flight, fid)
            try:
                services.hold_seat(db, f, "10C", u)
                results.append("won")
            except services.SeatUnavailable:
                results.append("lost")

    ts = [threading.Thread(target=worker, args=(u,)) for u in user_ids]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("won") == 1 and results.count("lost") == len(user_ids) - 1
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(SeatLock).where(SeatLock.flight_id == fid, SeatLock.seat_no == "10C")) == 1


def test_expired_hold_frees_the_seat(world):
    user_ids, fid = world
    with SessionLocal() as db:
        lock = db.scalar(select(SeatLock).where(SeatLock.flight_id == fid, SeatLock.seat_no == "10C"))
        lock.expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
        winner = db.get(User, user_ids[1])
        services.hold_seat(db, db.get(Flight, fid), "10C", winner)  # now succeeds
        assert db.scalar(select(SeatLock.booking_id).where(SeatLock.flight_id == fid, SeatLock.seat_no == "10C"))


def test_split_payment_confirms_only_after_last_share(world):
    user_ids, fid = world
    with SessionLocal() as db:
        u, f = db.get(User, user_ids[2]), db.get(Flight, fid)
        for s in ("11A", "11B", "11C"):
            services.hold_seat(db, f, s, u)
        b = services.pending_booking(db, u, fid)
        pays = services.prepare_checkout(db, b, {"11A": "A", "11B": "B", "11C": "C"}, 3)
        assert sum(p.amount for p in pays) == b.total_amount
        assert services.pay(db, pays[0], "card", "4242 4242 4242 4242", "A") == "pending"
        with pytest.raises(services.CheckoutError):
            services.pay(db, pays[1], "card", "4000 0000 0000 0002", "B")  # declined card
        assert services.pay(db, pays[1], "card", "4242424242424242", "B") == "pending"
        assert services.pay(db, pays[2], "gcash", "", "C") == "confirmed"
        assert db.scalar(select(func.count()).select_from(Passenger).where(Passenger.flight_id == fid, Passenger.seat_no.in_(["11A", "11B", "11C"]))) == 3
        # the seats are now permanently taken for everyone else
        with pytest.raises(services.SeatUnavailable):
            services.hold_seat(db, f, "11A", db.get(User, user_ids[3]))


def test_dijkstra_finds_connection_cheaper_or_equal_to_alternatives(world):
    with SessionLocal() as db:
        from app.models import Airport
        airports = {a.code: a for a in db.scalars(select(Airport))}
        day = (datetime.now() + timedelta(days=2)).date()
        start = datetime.combine(day, datetime.min.time())
        fl = db.scalars(select(Flight).where(Flight.depart >= start, Flight.depart < start + timedelta(days=2))).unique().all()
        price = {f.id: f.base_price for f in fl}
        its = routing.find_itineraries(fl, airports, "ILO", "DVO", day, price, max_stops=2)
        assert its and all(i.legs[0].origin == "ILO" and i.legs[-1].dest == "DVO" for i in its)
        cheapest = next(i for i in its if "Cheapest" in i.tags)
        assert cheapest.price == min(i.price for i in its)
        for i in its:  # every connection respects layover limits
            for lv in i.layovers:
                assert routing.MIN_LAYOVER <= lv["minutes"] <= routing.MAX_LAYOVER
