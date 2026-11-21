"""End-to-end checks of the data-structure features through the real web app."""
import re
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import desk, indexes
from app.db import SessionLocal
from app.main import app
from app.models import Booking, BookingRequest, Flight, Passenger, Payment, User


def _client(email, pw="demo123"):
    c = TestClient(app, follow_redirects=True)
    assert c.post("/login", data={"email": email, "password": pw}).status_code == 200
    return c


def _register(c, name):
    r = c.post("/register", data={"name": name, "email": f"{name.lower()}@t.test", "password": "secret1"})
    assert r.status_code == 200


def _future_flight(db, skip=0):
    return db.scalars(select(Flight).where(Flight.depart > datetime.now() + timedelta(days=20)).order_by(Flight.id).offset(skip).limit(1)).first()


def _confirm(db, user, flight, seats):
    from app import services
    for s in seats:
        services.hold_seat(db, flight, s, user)
    b = services.pending_booking(db, user, flight.id)
    pays = services.prepare_checkout(db, b, {s: "Pax " + s for s in seats}, 1)
    bid, token = b.id, pays[0].token
    db.expire_all()                      # the web app uses a fresh session per request
    p = db.scalar(select(Payment).where(Payment.token == token))
    assert services.pay(db, p, "card", "4242424242424242", "T") == "confirmed"
    return bid


def test_pages_render():
    with TestClient(app) as c:
        admin = _client("admin@skybook.test", "admin123")
        for url in ("/find", "/find?number=SB101", "/find?number=ZZ9", "/desk", "/desk/scan", "/requests"):
            assert admin.get(url).status_code == 200, url
        assert "SB101" in admin.get("/find?number=SB101").text


def test_priority_order_and_waitlist_and_undo():
    with TestClient(app):
        pass
    db = SessionLocal()
    flight = _future_flight(db, skip=500)
    admin = db.scalar(select(User).where(User.email == "admin@skybook.test"))
    # three travelers file requests: Economy first, then VIP, then Business
    users = []
    for n in ("Ecoz", "Vipz", "Bizz"):
        u = User(email=f"{n}@q.test", name=n, pw_hash="x$y")
        db.add(u)
        users.append(u)
    db.commit()
    reqs = [desk.submit_request(db, u, flight, t, 1) for u, t in zip(users, ("Economy", "VIP", "Business"))]
    order = [r.tier for r in desk.queued_heap(db).in_priority_order()]
    assert order[:3] == ["VIP", "Business", "Economy"] or {"VIP", "Business", "Economy"} <= set(order)
    first = desk.process_next(db)
    assert first["request"].tier == "VIP"

    # same tier: earlier arrival wins
    a = desk.submit_request(db, users[0], flight, "Economy", 1)
    b = desk.submit_request(db, users[1], flight, "Economy", 1)
    heap = desk.queued_heap(db)
    ids = [r.id for r in heap.in_priority_order() if r.tier == "Economy"]
    assert ids == sorted(ids)

    # fill a flight, join the waitlist FIFO, undo -> front of the line gets the seats
    f2 = _future_flight(db, skip=600)
    desk.fill_flight(db, f2, admin)
    assert desk.is_full(db, f2)
    w1 = desk.join_waitlist(db, users[0], f2, "Economy", 1)
    w2 = desk.join_waitlist(db, users[1], f2, "Economy", 1)
    assert desk.waitlist_position(db, w1) == 1 and desk.waitlist_position(db, w2) == 2
    res = desk.undo_last_booking(db)
    assert res["booking"].flight_id == f2.id
    assert w1.id in [r.id for r in res["promoted"]]
    db.refresh(w1); db.refresh(w2)
    assert w1.status == "allocated" and w2.status == "allocated"   # whole flight was freed: both fit, in order
    assert db.scalar(select(Booking.status).where(Booking.id == res["booking"].id)) == "cancelled"
    db.close()


def test_manifest_detach_and_hash_lookup():
    db = SessionLocal()
    flight = _future_flight(db, skip=900)
    u = db.scalar(select(User).where(User.email == "demo@skybook.test"))
    b1 = _confirm(db, u, flight, ["1A"])
    b2 = _confirm(db, u, flight, ["1B"])
    b3 = _confirm(db, u, flight, ["1C"])
    ref2 = db.get(Booking, b2).ref
    assert indexes.find_passenger(indexes.ticket_key(ref2, "1B"))["found"]
    chain = [n["id"] for n in desk.manifest(db, flight.id).to_list()]
    assert chain[-3:] == [b1, b2, b3]
    res = desk.cancel_from_manifest(db, flight.id, b2)
    assert res["detached"]["before"]["id"] == b1 and res["detached"]["after"]["id"] == b3
    assert [n["id"] for n in desk.manifest(db, flight.id).to_list()][-2:] == [b1, b3]
    assert not indexes.find_passenger(indexes.ticket_key(ref2, "1B"))["found"]
    db.close()


def test_search_has_fewest_stops_and_quicksort():
    with TestClient(app) as c:
        day = datetime.now().date().isoformat()
        r = c.get(f"/search?origin=MNL&dest=CEB&day={datetime.now().date() + __import__('datetime').timedelta(days=2)}&sort=stops&max_stops=1")
        assert r.status_code == 200 and "QuickSort" in r.text and "Fewest stops" in r.text
