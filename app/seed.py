"""Demo data: Philippine domestic network + HKG/SIN, flights for ~7 weeks, and
synthetic historical bookings so the admin dashboard has something to show."""
import hashlib
import os
import random
import secrets
from datetime import datetime, timedelta

from sqlalchemy import insert, select

from .models import (Aircraft, Airport, Booking, Flight, Passenger, Payment, User)
from .services import LETTERS, seat_class, seat_ids, seat_multiplier

AIRPORTS = [
    ("MNL", "Ninoy Aquino Intl", "Manila", 3), ("CEB", "Mactan-Cebu Intl", "Cebu", 3),
    ("DVO", "Francisco Bangoy Intl", "Davao", 3), ("CRK", "Clark Intl", "Clark", 2),
    ("ILO", "Iloilo Intl", "Iloilo", 2), ("BCD", "Bacolod-Silay", "Bacolod", 2),
    ("ZAM", "Zamboanga Intl", "Zamboanga", 2), ("PPS", "Puerto Princesa Intl", "Puerto Princesa", 1),
    ("KLO", "Kalibo Intl", "Kalibo (Boracay)", 1), ("TAG", "Tagbilaran", "Tagbilaran (Bohol)", 1),
    ("HKG", "Hong Kong Intl", "Hong Kong", 3), ("SIN", "Changi", "Singapore", 3),
]

# (a, b, flight minutes, economy base fare PHP)
ROUTES = [
    ("MNL", "CEB", 90, 2200), ("MNL", "DVO", 125, 3000), ("MNL", "CRK", 50, 1500), ("MNL", "ILO", 70, 1900),
    ("MNL", "BCD", 75, 1900), ("MNL", "PPS", 80, 2100), ("MNL", "KLO", 60, 1600), ("MNL", "ZAM", 110, 2800),
    ("MNL", "TAG", 65, 1700), ("MNL", "HKG", 135, 5200), ("MNL", "SIN", 215, 6400), ("CEB", "DVO", 80, 1900),
    ("CEB", "ILO", 55, 1500), ("CEB", "BCD", 45, 1300), ("CEB", "CRK", 100, 2400), ("CEB", "SIN", 235, 6800),
    ("CEB", "HKG", 170, 5600), ("DVO", "ZAM", 55, 1500),
]

FIRST = ["Juan", "Maria", "Jose", "Ana", "Mark", "Grace", "Paolo", "Liza", "Carlo", "Bea", "Miguel", "Nina",
         "Rico", "Tessa", "Gab", "Ines", "Luis", "Clara", "Noel", "Dana"]
LAST = ["Santos", "Reyes", "Cruz", "Garcia", "Ramos", "Mendoza", "Bautista", "Aquino", "Torres", "Flores",
        "Villanueva", "Castillo", "Navarro", "Domingo", "Aguilar"]


def hash_pw(pw: str) -> str:
    salt = secrets.token_hex(8)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()


def check_pw(pw: str, stored: str) -> bool:
    salt, _, h = stored.partition("$")
    return secrets.compare_digest(hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex(), h)


def seed(db, history_days_back=7, history_days_fwd=7, days_ahead=45):
    if db.scalar(select(Flight.id).limit(1)):
        return
    rng = random.Random(2026)
    db.add_all([Airport(code=c, name=n, city=ci, size=s) for c, n, ci, s in AIRPORTS])
    atr = Aircraft(model="ATR 72-600", rows=14, cols=4, business_rows=0)
    a320 = Aircraft(model="Airbus A320", rows=20, cols=6, business_rows=2)
    db.add_all([atr, a320])
    db.add_all([
        User(email="admin@skybook.test", name="Airline Admin", pw_hash=hash_pw(os.getenv("ADMIN_PASSWORD", "admin123")), is_admin=True),
        User(email="demo@skybook.test", name="Demo Traveler", pw_hash=hash_pw("demo123")),
    ])
    db.commit()
    small = {a for a, _, _, s in AIRPORTS if s == 1}

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    flights = []
    n = 100
    for i, (a, b, mins, base) in enumerate(ROUTES):
        ac = atr if (a in small or b in small) and mins < 100 else a320
        slots = [(6 + (i * 3) % 5, (i * 15) % 60), (14 + (i * 2) % 5, (i * 25 + 10) % 60)]
        for d in range(-history_days_back, days_ahead + 1):
            for si, (h, m) in enumerate(slots):
                for (o, de, off) in ((a, b, 0), (b, a, 70)):  # return flight ~70 min later in the day
                    dep = today + timedelta(days=d, hours=h, minutes=m + off)
                    n += 1
                    flights.append(Flight(number=f"SB{n}", origin=o, dest=de, depart=dep,
                                          arrive=dep + timedelta(minutes=mins),
                                          base_price=int(round(base * rng.uniform(0.92, 1.1) / 10) * 10),
                                          aircraft_id=ac.id))
    db.add_all(flights)
    db.commit()

    # ---- synthetic booking history (bulk insert) ----
    demo = db.scalar(select(User).where(User.email == "demo@skybook.test"))
    now = datetime.now()
    near = [f for f in flights if now - timedelta(days=history_days_back) <= f.depart <= now + timedelta(days=history_days_fwd)]
    bookings, passengers, payments = [], [], []
    bid = pid = payid = 0
    for f in near:
        ac = a320 if f.aircraft_id == a320.id else atr
        ahead = max((f.depart - now).days, 0)
        load = rng.uniform(0.45, 0.92) if f.depart < now else rng.uniform(0.15, 0.7) * (1 - ahead / 14)
        want = int(ac.capacity * load)
        seats = rng.sample(seat_ids(ac), min(want, ac.capacity))
        i = 0
        while i < len(seats):
            group = seats[i:i + rng.randint(1, 4)]
            i += len(group)
            bid += 1
            created = min(f.depart - timedelta(days=rng.uniform(1, 25)), now - timedelta(minutes=5))
            total = 0
            for s in group:
                price = int(round(f.base_price * rng.uniform(0.95, 1.5) * seat_multiplier(ac, int(s[:-1])) / 10) * 10)
                total += price
                pid += 1
                passengers.append(dict(id=pid, booking_id=bid, flight_id=f.id, seat_no=s, price=price,
                                       name=f"{rng.choice(FIRST)} {rng.choice(LAST)}"))
            bookings.append(dict(id=bid, ref=f"H{bid:05d}", user_id=demo.id, flight_id=f.id, status="confirmed",
                                 total_amount=total, created_at=created))
            payid += 1
            payments.append(dict(id=payid, booking_id=bid, token=f"seed{payid:07d}", label="Payer 1", amount=total,
                                 status="paid", method="card", last4="4242", provider_ref="seed", payer_name="Seed",
                                 paid_at=created))
    for chunk_src, model in ((bookings, Booking), (passengers, Passenger), (payments, Payment)):
        for k in range(0, len(chunk_src), 5000):
            db.execute(insert(model), chunk_src[k:k + 5000])
    db.commit()
