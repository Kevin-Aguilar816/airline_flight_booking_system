"""Flight In, Flight Out - flight booking system (FastAPI + SQLAlchemy + HTMX + Tailwind).

Run:  uvicorn app.main:app --reload      then open http://127.0.0.1:8000
"""
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, literal_column, select
from starlette.middleware.sessions import SessionMiddleware

from . import desk, indexes, ml, routing, services
from .dsa import quicksort
from .db import Base, SessionLocal, engine, get_db, utcnow
from .models import Airport, Booking, BookingRequest, Flight, Passenger, Payment, User
from .seed import check_pw, hash_pw, seed
from .ticket import build_ticket_pdf

BASE = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(_app):
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)
        indexes.build_all(db)  # BST of flights, hash table of passengers, undo stack
    ml.get_model()  # train / load the price predictor at boot
    yield


app = FastAPI(title="Flight In, Flight Out", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev-only-change-me"), same_site="lax")
templates = Jinja2Templates(directory=str(BASE / "templates"))
templates.env.filters["peso"] = lambda v: f"₱{int(v):,}"
templates.env.filters["hm"] = lambda m: f"{int(m) // 60}h {int(m) % 60:02d}m"
templates.env.filters["tm"] = lambda d: d.strftime("%H:%M")
templates.env.filters["dt"] = lambda d: d.strftime("%a, %d %b %Y %H:%M")
templates.env.filters["day"] = lambda d: d.strftime("%a, %d %b %Y")
templates.env.filters["iso_utc"] = lambda d: d.isoformat() + "Z" if d else ""


# ------------------------------------------------------------------ auth ----
class NeedLogin(Exception):
    def __init__(self, next_url: str):
        self.next_url = next_url


@app.exception_handler(NeedLogin)
async def _need_login(request: Request, exc: NeedLogin):
    url = f"/login?next={exc.next_url}"
    if request.headers.get("HX-Request"):
        return Response(status_code=200, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


def current_user(request: Request, db=Depends(get_db)) -> User | None:
    uid = request.session.get("uid")
    return db.get(User, uid) if uid else None


def require_user(request: Request, user=Depends(current_user)) -> User:
    if not user:
        raise NeedLogin(request.url.path)
    return user


def require_admin(user=Depends(require_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admins only")
    return user


def flash(request: Request, text: str, ok: bool = True):
    request.session["flash"] = {"text": text, "ok": ok}


def page(request: Request, name: str, user, status_code=200, **ctx):
    note = None if name.startswith("_") else request.session.pop("flash", None)  # not for HTMX fragments
    return templates.TemplateResponse(request, name, {"user": user, "flash": note, **ctx}, status_code=status_code)


def message(request, user, title, text, status_code=200, link=("/", "Back to search")):
    return page(request, "message.html", user, status_code, title=title, text=text, link=link)


def safe_next(url: str) -> str:
    return url if url.startswith("/") and not url.startswith("//") else "/"


# ----------------------------------------------------------------- pages ----
@app.get("/", response_class=HTMLResponse)
def home(request: Request, db=Depends(get_db), user=Depends(current_user)):
    airports = db.scalars(select(Airport).order_by(Airport.city)).all()
    return page(request, "index.html", user, airports=airports, tomorrow=(date.today() + timedelta(days=1)).isoformat())


def _airports(db) -> dict[str, Airport]:
    return {a.code: a for a in db.scalars(select(Airport))}


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, origin: str, dest: str, day: str, max_stops: int = 1, sort: str = "price",
           db=Depends(get_db), user=Depends(current_user)):
    try:
        d = date.fromisoformat(day)
    except ValueError:
        return message(request, user, "Invalid date", "Please pick a valid travel date.", 400)
    airports = _airports(db)
    if origin == dest or origin not in airports or dest not in airports:
        return message(request, user, "Check your route", "Pick two different, valid airports.", 400)
    start = datetime.combine(d, time.min)
    window = db.scalars(select(Flight).where(Flight.depart >= start, Flight.depart < start + timedelta(days=2),
                                             Flight.depart > datetime.now())).unique().all()
    loads = services.loads_for(db, window)
    price_of = {f.id: services.dynamic_fare(f, loads[f.id]) for f in window}
    its = routing.find_itineraries(window, airports, origin, dest, d, price_of, max_stops=max(0, min(max_stops, 2)))

    demand = airports[origin].size + airports[dest].size
    for it in its:
        preds = [ml.predict(max((f.depart - datetime.now()).days, 0), loads[f.id], f.depart,
                            airports[f.origin].size + airports[f.dest].size, f.base_price) for f in it.legs]
        p = sum(x["p_drop"] for x in preds) / len(preds)
        it.advice = {"advice": "Wait" if p >= .5 else "Buy Now", "confidence": round(max(p, 1 - p) * 100)}
        it.seats_left = min(f.aircraft.capacity - round(loads[f.id] * f.aircraft.capacity) for f in it.legs)
    keys = {"price": lambda i: (i.price, i.minutes), "duration": lambda i: (i.minutes, i.price),
            "risk": lambda i: (i.risk_score, i.price), "stops": lambda i: (i.stops, i.price)}
    its, sort_comparisons = quicksort(its, key=keys.get(sort, keys["price"]))  # our own QuickSort
    return page(request, "results.html", user, its=its, o=airports[origin], d=airports[dest], day=d, sort=sort,
                max_stops=max_stops, query=f"origin={origin}&dest={dest}&day={day}", demand=demand,
                sort_comparisons=sort_comparisons)


# ------------------------------------------------------------ seat select ---
def _flight(db, fid) -> Flight:
    f = db.get(Flight, fid)
    if not f:
        raise HTTPException(404, "Flight not found")
    return f


@app.get("/flights/{fid}", response_class=HTMLResponse)
def flight_page(fid: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    f = _flight(db, fid)
    loads = services.loads_for(db, [f])
    pred = ml.predict(max((f.depart - datetime.now()).days, 0), loads[f.id], f.depart,
                      f.origin_airport.size + f.dest_airport.size, f.base_price)
    return page(request, "seats.html", user, f=f, sm=services.build_seatmap(db, f, user), msg=None, pred=pred,
                hold_minutes=services.HOLD_MINUTES, tiers=list(desk.TIERS), full=desk.is_full(db, f),
                waiting=len(desk.waitlist_deque(db, f.id)))


def _panel(request, db, f, user, msg=None, ok=False):
    return page(request, "_seatpanel.html", user, f=f, sm=services.build_seatmap(db, f, user), msg=msg, ok=ok)


@app.get("/flights/{fid}/panel", response_class=HTMLResponse)
def panel(fid: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    return _panel(request, db, _flight(db, fid), user)


@app.post("/flights/{fid}/seats/{seat}/hold", response_class=HTMLResponse)
def hold(fid: int, seat: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
    f = _flight(db, fid)
    try:
        services.hold_seat(db, f, seat, user)
        return _panel(request, db, f, user, f"Seat {seat} held for {services.HOLD_MINUTES} minutes.", ok=True)
    except services.SeatUnavailable as e:
        return _panel(request, db, f, user, str(e))


@app.post("/flights/{fid}/seats/{seat}/release", response_class=HTMLResponse)
def release(fid: int, seat: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
    f = _flight(db, fid)
    try:
        services.release_seat(db, f, seat, user)
    except services.SeatUnavailable as e:
        return _panel(request, db, f, user, str(e))
    return _panel(request, db, f, user, f"Seat {seat} released.", ok=True)


# ---------------------------------------------------------------- checkout --
def _my_booking(db, ref, user) -> Booking:
    b = db.scalar(select(Booking).where(Booking.ref == ref))
    if not b or (b.user_id != user.id and not user.is_admin):
        raise HTTPException(404, "Booking not found")
    return b


@app.get("/bookings/{ref}/checkout", response_class=HTMLResponse)
def checkout_form(ref: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
    b = _my_booking(db, ref, user)
    if b.status != "pending":
        return RedirectResponse(f"/bookings/{ref}", 303)
    locks = services.active_locks(db, b)
    if not locks:
        return message(request, user, "Seat hold expired", "Your 10-minute hold ended. Please choose your seats again.",
                       link=(f"/flights/{b.flight_id}", "Back to seat map"))
    return page(request, "checkout.html", user, b=b, locks=locks, total=sum(l.price for l in locks), error=None,
                expires=min(l.expires_at for l in locks))


@app.post("/bookings/{ref}/checkout", response_class=HTMLResponse)
async def checkout_submit(ref: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
    b = _my_booking(db, ref, user)
    form = await request.form()
    names = {k[5:]: str(v) for k, v in form.items() if k.startswith("name_")}
    try:
        services.prepare_checkout(db, b, names, int(form.get("payers", 1)))
    except services.CheckoutError as e:
        locks = services.active_locks(db, b)
        if not locks:
            return message(request, user, "Seat hold expired", str(e), link=(f"/flights/{b.flight_id}", "Back to seat map"))
        return page(request, "checkout.html", user, b=b, locks=locks, total=sum(l.price for l in locks), error=str(e),
                    expires=min(l.expires_at for l in locks), status_code=400)
    return RedirectResponse(f"/bookings/{ref}", 303)


# ----------------------------------------------------------- group payment --
def _payment(db, token) -> Payment:
    p = db.scalar(select(Payment).where(Payment.token == token))
    if not p:
        raise HTTPException(404, "Payment link not found")
    return p


@app.get("/pay/{token}", response_class=HTMLResponse)
def pay_form(token: str, request: Request, db=Depends(get_db), user=Depends(current_user)):
    p = _payment(db, token)
    locks = services.active_locks(db, p.booking) if p.booking.status == "pending" else []
    return page(request, "pay.html", user, p=p, b=p.booking, error=None, expires=min((l.expires_at for l in locks), default=None))


@app.post("/pay/{token}", response_class=HTMLResponse)
def pay_submit(token: str, request: Request, method: str = Form("card"), card: str = Form(""),
               payer_name: str = Form(""), db=Depends(get_db), user=Depends(current_user)):
    p = _payment(db, token)
    try:
        status = services.pay(db, p, method, card, payer_name)
    except services.CheckoutError as e:
        locks = services.active_locks(db, p.booking) if p.booking.status == "pending" else []
        return page(request, "pay.html", user, status_code=400, p=p, b=p.booking, error=str(e),
                    expires=min((l.expires_at for l in locks), default=None))
    return page(request, "paid.html", user, p=p, b=db.get(Booking, p.booking_id), status=status)


# ----------------------------------------------------------------- bookings -
@app.get("/bookings", response_class=HTMLResponse)
def my_bookings(request: Request, db=Depends(get_db), user=Depends(require_user)):
    rows = db.scalars(select(Booking).where(Booking.user_id == user.id, Booking.status != "pending")
                      .order_by(Booking.created_at.desc()).limit(50)).unique().all()
    pending = db.scalars(select(Booking).where(Booking.user_id == user.id, Booking.status == "pending")).unique().all()
    return page(request, "bookings.html", user, rows=rows, pending=[b for b in pending if services.active_locks(db, b)])


@app.get("/bookings/{ref}", response_class=HTMLResponse)
def booking_detail(ref: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
    b = _my_booking(db, ref, user)
    locks = services.active_locks(db, b) if b.status == "pending" else []
    return page(request, "booking.html", user, b=b, locks=locks, base=str(request.base_url).rstrip("/"),
                expires=min((l.expires_at for l in locks), default=None))


@app.get("/bookings/{ref}/ticket.pdf")
def ticket(ref: str, db=Depends(get_db), user=Depends(require_user)):
    b = _my_booking(db, ref, user)
    if b.status != "confirmed":
        raise HTTPException(409, "Ticket available after payment is complete")
    return Response(build_ticket_pdf(b), media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="ticket-{b.ref}.pdf"'})


# --------------------------------------------------------------- auth pages -
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/", user=Depends(current_user)):
    return page(request, "login.html", user, next=next, error=None, mode="login")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), next: str = Form("/"), db=Depends(get_db)):
    u = db.scalar(select(User).where(User.email == email.strip().lower()))
    if not u or not check_pw(password, u.pw_hash):
        return page(request, "login.html", None, status_code=400, next=next, error="Wrong email or password.", mode="login")
    request.session["uid"] = u.id
    return RedirectResponse(safe_next(next), 303)


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request, user=Depends(current_user)):
    return page(request, "login.html", user, next="/", error=None, mode="register")


@app.post("/register")
def register(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...),
             db=Depends(get_db)):
    email = email.strip().lower()
    if len(password) < 6 or "@" not in email:
        return page(request, "login.html", None, status_code=400, next="/", mode="register",
                    error="Use a valid email and a password of at least 6 characters.")
    if db.scalar(select(User.id).where(User.email == email)):
        return page(request, "login.html", None, status_code=400, next="/", mode="register", error="Email already registered.")
    u = User(email=email, name=name.strip()[:120], pw_hash=hash_pw(password))
    db.add(u)
    db.commit()
    request.session["uid"] = u.id
    return RedirectResponse("/", 303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", 303)


# --------------------------------------------------------- admin dashboard --
@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, db=Depends(get_db), user=Depends(require_admin)):
    confirmed = Booking.status == "confirmed"
    since = utcnow() - timedelta(days=14)
    rev_rows = db.execute(select(func.date(Booking.created_at), func.sum(Booking.total_amount))
                          .where(confirmed, Booking.created_at >= since).group_by(func.date(Booking.created_at))
                          .order_by(func.date(Booking.created_at))).all()
    route = (Flight.origin + literal_column("'-'") + Flight.dest)
    pop = db.execute(select(route, func.count(Passenger.id)).join(Flight, Flight.id == Passenger.flight_id)
                     .group_by(route).order_by(func.count(Passenger.id).desc()).limit(8)).all()
    # load factor per route = passengers / seats offered on flights that have passengers
    pax = dict(db.execute(select(Passenger.flight_id, func.count()).group_by(Passenger.flight_id)).all())
    fl = db.scalars(select(Flight).where(Flight.id.in_(list(pax)))).unique().all()
    agg: dict[str, list[int]] = {}
    for f in fl:
        a = agg.setdefault(f"{f.origin}-{f.dest}", [0, 0])
        a[0] += pax[f.id]
        a[1] += f.aircraft.capacity
    lf = sorted(((r, round(100 * p / c, 1)) for r, (p, c) in agg.items()), key=lambda x: -x[1])[:8]
    total_rev = db.scalar(select(func.coalesce(func.sum(Booking.total_amount), 0)).where(confirmed))
    total_pax = db.scalar(select(func.count(Passenger.id)))
    total_book = db.scalar(select(func.count(Booking.id)).where(confirmed))
    cap = sum(f.aircraft.capacity for f in fl)
    kpis = {"revenue": total_rev, "bookings": total_book, "pax": total_pax,
            "load": round(100 * total_pax / cap, 1) if cap else 0,
            "avg": round(total_rev / total_pax) if total_pax else 0}
    charts = {"rev": {"labels": [str(d) for d, _ in rev_rows], "data": [int(v) for _, v in rev_rows]},
              "pop": {"labels": [r for r, _ in pop], "data": [c for _, c in pop]},
              "lf": {"labels": [r for r, _ in lf], "data": [v for _, v in lf]}}
    return page(request, "admin.html", user, kpis=kpis, charts=json.dumps(charts), ml=ml.metrics)


# =============================================================================
# Data-structure features: flight-number search (BST), priority requests (heap),
# waitlist (deque), undo (stack), passenger scanner (hash table),
# manifest (linked list). Logic lives in desk.py / indexes.py / dsa.py.
# =============================================================================
@app.get("/find", response_class=HTMLResponse)
def find_flight(request: Request, number: str = "", db=Depends(get_db), user=Depends(current_user)):
    result = indexes.find_flight(number) if number.strip() else None
    flight = db.get(Flight, result["value"]) if result and result["found"] else None
    return page(request, "find.html", user, number=number.strip().upper(), r=result, f=flight,
                stats=indexes.tree_stats())


@app.post("/flights/{fid}/request")
def priority_request(fid: int, request: Request, tier: str = Form(...), seats: int = Form(1),
                     db=Depends(get_db), user=Depends(require_user)):
    try:
        desk.submit_request(db, user, _flight(db, fid), tier, seats)
        flash(request, "Request sent to the booking desk. VIP is served first, then Business, then Economy.")
    except desk.DeskError as e:
        flash(request, str(e), ok=False)
        return RedirectResponse(f"/flights/{fid}", 303)
    return RedirectResponse("/requests", 303)


@app.post("/flights/{fid}/waitlist")
def waitlist_join(fid: int, request: Request, tier: str = Form(...), seats: int = Form(1),
                  db=Depends(get_db), user=Depends(require_user)):
    try:
        desk.join_waitlist(db, user, _flight(db, fid), tier, seats)
        flash(request, "You joined the waitlist. If seats free up you are offered them first-come, first-served.")
    except desk.DeskError as e:
        flash(request, str(e), ok=False)
        return RedirectResponse(f"/flights/{fid}", 303)
    return RedirectResponse("/requests", 303)


@app.get("/requests", response_class=HTMLResponse)
def my_requests(request: Request, db=Depends(get_db), user=Depends(require_user)):
    desk.expire_offers(db)
    rows = db.scalars(select(BookingRequest).where(BookingRequest.user_id == user.id)
                      .order_by(BookingRequest.id.desc()).limit(50)).unique().all()
    refs = {r.id: db.get(Booking, r.booking_id).ref for r in rows if r.booking_id}
    pos = {r.id: desk.waitlist_position(db, r) for r in rows if r.status == "waitlisted"}
    return page(request, "requests.html", user, rows=rows, refs=refs, pos=pos)


@app.post("/requests/{rid}/cancel")
def request_cancel(rid: int, request: Request, db=Depends(get_db), user=Depends(require_user)):
    req = db.get(BookingRequest, rid)
    if not req:
        raise HTTPException(404, "Request not found")
    try:
        desk.cancel_request(db, req, user)
        flash(request, "Request withdrawn.")
    except desk.DeskError as e:
        flash(request, str(e), ok=False)
    return RedirectResponse("/requests", 303)


# ------------------------------------------------------------- agent desk ---
@app.get("/desk", response_class=HTMLResponse)
def desk_page(request: Request, db=Depends(get_db), user=Depends(require_admin)):
    desk.expire_offers(db)
    heap = desk.queued_heap(db)
    undo_ids = indexes.undo_stack.top_first(8)
    undo = []
    for bid in undo_ids:
        b = db.get(Booking, bid)
        if b:
            undo.append({"b": b, "seats": sorted(p.seat_no for p in b.passengers)})
    return page(request, "desk.html", user, queue=heap.in_priority_order(), heap_array=heap.array(),
                undo=undo, undo_size=len(indexes.undo_stack), waitlists=desk.all_waitlists(db),
                tree=indexes.tree_stats(), table=indexes.table_stats(), tiers=desk.TIERS)


@app.post("/desk/process-next")
def desk_process_next(request: Request, db=Depends(get_db), user=Depends(require_admin)):
    r = desk.process_next(db)
    if not r:
        flash(request, "The priority queue is empty.", ok=False)
    elif r["action"] == "allocated":
        flash(request, f"Processed {r['request'].tier} request #{r['request'].id} ({r['request'].user.name}): "
                       f"seats {', '.join(sorted(r['seats']))} held for them.")
    else:
        flash(request, f"Request #{r['request'].id} ({r['request'].tier}) had no seats: moved to the waitlist.", ok=False)
    return RedirectResponse("/desk", 303)


@app.post("/desk/process-all")
def desk_process_all(request: Request, db=Depends(get_db), user=Depends(require_admin)):
    order = []
    while len(order) < 100:
        r = desk.process_next(db)
        if not r:
            break
        order.append(f"#{r['request'].id} {r['request'].tier} ({'seats' if r['action'] == 'allocated' else 'waitlisted'})")
    flash(request, "Processed in priority order: " + (", ".join(order) if order else "nothing queued."), ok=bool(order))
    return RedirectResponse("/desk", 303)


def _undo_message(res: dict, prefix: str) -> str:
    text = f"{prefix} booking {res['booking'].ref} (seats {', '.join(res['seats'])}) cancelled and refunded."
    if res["promoted"]:
        text += " Waitlist promoted from the front: " + ", ".join(
            f"{r.user.name} ({r.seats} seat{'s' if r.seats > 1 else ''})" for r in res["promoted"]) + "."
    else:
        text += " No one on the waitlist was ready for those seats."
    return text


@app.post("/desk/undo")
def desk_undo(request: Request, db=Depends(get_db), user=Depends(require_admin)):
    try:
        flash(request, _undo_message(desk.undo_last_booking(db), "Undo: most recent"))
    except desk.DeskError as e:
        flash(request, str(e), ok=False)
    return RedirectResponse("/desk", 303)


@app.post("/desk/fill")
def desk_fill(request: Request, number: str = Form(...), db=Depends(get_db), user=Depends(require_admin)):
    hit = indexes.find_flight(number)
    if not hit["found"]:
        flash(request, f"No flight {number.upper()}.", ok=False)
    else:
        try:
            n = desk.fill_flight(db, _flight(db, hit["value"]), user)
            flash(request, f"Flight {number.upper()} filled for the demo ({n} seats sold). It can now take waitlist joins.")
        except desk.DeskError as e:
            flash(request, str(e), ok=False)
    return RedirectResponse("/desk", 303)


@app.get("/desk/manifest-find")
def desk_manifest_find(request: Request, number: str = "", user=Depends(require_admin)):
    hit = indexes.find_flight(number)
    if not hit["found"]:
        flash(request, f"No flight {number.strip().upper()}.", ok=False)
        return RedirectResponse("/desk", 303)
    return RedirectResponse(f"/desk/flights/{hit['value']}/manifest", 303)


@app.get("/desk/flights/{fid}/manifest", response_class=HTMLResponse)
def desk_manifest(fid: int, request: Request, db=Depends(get_db), user=Depends(require_admin)):
    f = _flight(db, fid)
    return page(request, "manifest.html", user, f=f, chain=desk.manifest(db, fid).to_list(),
                waiting=len(desk.waitlist_deque(db, fid)))


@app.post("/desk/flights/{fid}/manifest/{bid}/cancel")
def desk_manifest_cancel(fid: int, bid: int, request: Request, db=Depends(get_db), user=Depends(require_admin)):
    try:
        res = desk.cancel_from_manifest(db, fid, bid)
        d = res["detached"]
        link = f"Detached {d['value']['ref']} from the chain: {d['before']['ref'] if d['before'] else 'HEAD'} now points to " \
               f"{d['after']['ref'] if d['after'] else 'END'}. "
        flash(request, link + _undo_message(res, "Cancelled"))
    except desk.DeskError as e:
        flash(request, str(e), ok=False)
    return RedirectResponse(f"/desk/flights/{fid}/manifest", 303)


@app.get("/desk/scan", response_class=HTMLResponse)
def desk_scan(request: Request, key: str = "", db=Depends(get_db), user=Depends(require_admin)):
    r = indexes.find_passenger(key) if key.strip() else None
    pax = db.get(Passenger, r["value"]) if r and r["found"] else None
    booking = db.get(Booking, pax.booking_id) if pax else None
    samples = []
    for bid in indexes.undo_stack.top_first(4):
        b = db.get(Booking, bid)
        if b and b.status == "confirmed":
            samples += [indexes.ticket_key(b.ref, p.seat_no) for p in b.passengers[:2]]
    return page(request, "scan.html", user, key=key.strip().upper(), r=r, pax=pax, booking=booking,
                table=indexes.table_stats(), samples=samples[:4], ticket_key=indexes.ticket_key)


# ---------------------------------------------------------------- JSON API --
@app.get("/api/route")
def api_route(origin: str, dest: str, day: str, max_stops: int = 2, db=Depends(get_db)):
    """Route optimizer as an API (see /docs for the interactive Swagger UI)."""
    d = date.fromisoformat(day)
    airports = _airports(db)
    start = datetime.combine(d, time.min)
    window = db.scalars(select(Flight).where(Flight.depart >= start, Flight.depart < start + timedelta(days=2))).unique().all()
    loads = services.loads_for(db, window)
    price_of = {f.id: services.dynamic_fare(f, loads[f.id]) for f in window}
    its = routing.find_itineraries(window, airports, origin, dest, d, price_of, max_stops=max_stops)
    return [{"legs": [f"{f.number} {f.origin}->{f.dest} {f.depart:%Y-%m-%d %H:%M}" for f in i.legs],
             "price": i.price, "minutes": i.minutes, "tags": i.tags, "risk": i.risk_level,
             "layovers": i.layovers} for i in its]


@app.get("/api/predict/{fid}")
def api_predict(fid: int, db=Depends(get_db)):
    f = _flight(db, fid)
    load = services.loads_for(db, [f])[f.id]
    return ml.predict(max((f.depart - datetime.now()).days, 0), load, f.depart,
                      f.origin_airport.size + f.dest_airport.size, f.base_price)
