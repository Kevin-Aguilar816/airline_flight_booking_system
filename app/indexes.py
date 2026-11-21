"""In-memory indexes that sit in front of the database.

The database stays the source of truth. These structures are rebuilt at start-up
and kept in step when bookings are confirmed or cancelled:

  flight_tree      BST        flight number  -> flight id
  passenger_table  HashTable  ticket key     -> passenger id   (key = BOOKINGREF-SEAT)
  undo_stack       Stack      ids of confirmed bookings, most recent on top

A re-entrant lock protects them because FastAPI runs sync routes in a thread pool.
(With several worker processes each would hold its own copy; this project runs
one process, which is also what the Dockerfile does.)
"""
import threading

from sqlalchemy import select

from .dsa import BST, HashTable, Stack
from .models import Booking, Flight, Passenger, Payment

lock = threading.RLock()
flight_tree: BST = BST()
passenger_table: HashTable = HashTable(1024)
undo_stack: Stack = Stack()


def ticket_key(ref: str, seat: str) -> str:
    """The unique 'digital key' printed on the boarding pass."""
    return f"{ref}-{seat}".upper()


def build_all(db) -> dict:
    global flight_tree, passenger_table, undo_stack
    with lock:
        # 1) BST of flights, built balanced from the sorted flight numbers
        pairs = [(n, i) for n, i in db.execute(select(Flight.number, Flight.id).order_by(Flight.number))]
        flight_tree = BST.build_balanced(pairs)

        # 2) hash table of every confirmed passenger
        table = HashTable(1024)
        rows = db.execute(select(Booking.ref, Passenger.seat_no, Passenger.id)
                          .join(Passenger, Passenger.booking_id == Booking.id))
        for ref, seat, pid in rows:
            table.put(ticket_key(ref, seat), pid)
        passenger_table = table

        # 3) undo stack: bookings made through the app (not the seeded history), oldest first
        seeded = select(Payment.booking_id).where(Payment.provider_ref == "seed")
        ids = db.scalars(select(Booking.id).where(Booking.status == "confirmed", Booking.id.not_in(seeded))
                         .order_by(Booking.id)).all()
        stack = Stack()
        for bid in ids[-200:]:
            stack.push(bid)
        undo_stack = stack
        return {"flights": len(flight_tree), "passengers": len(passenger_table), "undo": len(undo_stack)}


def on_confirmed(db, booking: Booking) -> None:
    with lock:
        undo_stack.push(booking.id)
        for p in booking.passengers:
            passenger_table.put(ticket_key(booking.ref, p.seat_no), p.id)


def forget_passengers(keys: list[str]) -> None:
    with lock:
        for k in keys:
            passenger_table.remove(k)


def find_flight(number: str) -> dict:
    with lock:
        return flight_tree.search(number.strip().upper())


def find_passenger(key: str) -> dict:
    with lock:
        return passenger_table.get(key.strip().upper())


def tree_stats() -> dict:
    with lock:
        return {"flights": len(flight_tree), "height": flight_tree.height()}


def table_stats() -> dict:
    with lock:
        return passenger_table.stats()
