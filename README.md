# Flight In, Flight Out — Flight Booking System (Capstone)

FastAPI · SQLAlchemy · PostgreSQL/SQLite · HTMX · Tailwind · scikit-learn · ReportLab · Chart.js

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # http://127.0.0.1:8000  (SQLite, zero setup)
```

Or with PostgreSQL (recommended for the defense):

```bash
docker compose up --build              # http://localhost:8000
# or: DATABASE_URL=postgresql://user:pass@host/db uvicorn app.main:app
```

First start creates the schema and seeds ~2,500 flights (12 airports), demo bookings and users:

| Role      | Login                    |
|-----------|--------------------------|
| Traveler  | demo@skybook.test / demo123 |
| Admin     | admin@skybook.test / admin123 |

Interactive API docs: `/docs`. Tests: `pytest -q`.
Sandbox cards: `4242 4242 4242 4242` approves · `4000 0000 0000 0002` declines · `4000 0000 0000 9995` insufficient funds. GCash is simulated.

## Phases → code

| Phase | What | Where |
|---|---|---|
| 1 | Users, airports, aircraft seat maps, flights, bookings, passengers | `app/models.py`, `app/seed.py` |
| 2 | Search, seat map, concurrency-safe holds | `app/main.py`, `app/services.py` |
| 3 | Route optimizer, ML predictor, layover risk, split checkout, sandbox payments, PDF e-tickets | `routing.py`, `ml.py`, `payments.py`, `ticket.py` |
| 4 | Admin dashboard (revenue, load factor, popular routes) | `/admin`, `templates/admin.html` |

## The four innovative features

**1. Graph route optimizer (`routing.py`)** — time-expanded graph: nodes are *flights*, edges connect flight A→B when A.dest = B.origin and the layover is 60 min–12 h. Dijkstra (binary heap, O((V+E) log V)) runs once per objective: price, duration, and "safe" (price + 15 × risk score). Popping more goal states yields next-best alternatives; results are tagged Cheapest / Fastest / Safest.

**2. ML Buy Now / Wait (`ml.py`)** — RandomForest predicts P(fare drops ≥5 % within 7 days) from days-to-departure, load factor, weekday, month, route demand, base fare. **The training data is synthetic** (no historical fare dataset is bundled); say so in your defense, and swap `synthetic_dataset()` for real data with the same columns.

**3. Concurrency-safe seat locking + split payment (`services.py`)** — a hold is an `INSERT` into `seat_locks` with `UNIQUE(flight_id, seat_no)`; the database lets exactly one concurrent insert win, the rest get `IntegrityError` → "seat taken". Holds last 10 min (`HOLD_MINUTES`) and expired holds are purged on the next attempt. The price is frozen at hold time. Checkout splits the total across N payers, each with their own `/pay/<token>` link; when the last share is paid, `finalize_if_paid` converts holds into passengers in one transaction (`SELECT … FOR UPDATE` on PostgreSQL) and the `UNIQUE(flight_id, seat_no)` on `passengers` is the final guard. If the hold lapses before everyone pays, everything is refunded. `tests/test_seat_locking.py` fires 25 simultaneous users at one seat: exactly 1 wins.

**4. Layover risk engine (`routing.py::layover_risk`)** — score = buffer after expected inbound delay + airport size + weather + delay history → Low / Medium / High. Weather and delays are **simulated deterministically** (hash of airport/flight/date); plug a real weather/flight-status API into `simulated_weather` / `expected_inbound_delay`.

## Known limits (be upfront about these)
- Payments are sandbox only; `payments.py` defines the gateway interface where a Stripe/PayMongo adapter would go (not implemented).
- Connecting itineraries are booked leg by leg (one booking per flight), not as a single PNR.
- Weather/delay data and the ML training set are synthetic.
- Set a real `SECRET_KEY` and change the seeded admin password before any public deployment.
