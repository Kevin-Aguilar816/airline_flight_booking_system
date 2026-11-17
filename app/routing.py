"""Graph-based multi-city route optimizer + layover risk engine.

Graph model (time-expanded):
  * node  = a scheduled FLIGHT (not just an airport) - so connections are
            always time-feasible
  * edge  = flight A -> flight B  iff  A.dest == B.origin and the layover is
            between MIN_LAYOVER and MAX_LAYOVER minutes
  * edge weight depends on the objective:
        price    -> fare of B
        duration -> layover + flight time of B
        safe     -> fare of B + RISK_PESO * missed-connection risk score

Dijkstra with a binary heap: O((V + E) log V). All weights are >= 0, so the
first time a goal node is popped its path is optimal for that objective.
Continuing to pop yields the next-best alternatives.
"""
import hashlib
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

MIN_LAYOVER = 60       # minutes
MAX_LAYOVER = 12 * 60
RISK_PESO = 15         # PHP of "pain" per risk point in the 'safe' objective


# --------------------------------------------------------------------------
# Layover risk scoring (rule-based heuristic)
# --------------------------------------------------------------------------
def _unit(*parts) -> float:
    """Deterministic pseudo-random number in [0,1) - stands in for live
    weather / delay APIs so the demo is reproducible."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2 ** 32


def simulated_weather(airport: str, day: date) -> tuple[float, str]:
    sev = _unit("wx", airport, day.isoformat()) ** 3  # skewed towards calm weather
    label = ("Clear" if sev < .12 else "Showers" if sev < .35
             else "Thunderstorms" if sev < .6 else "Typhoon warning")
    return sev, label


def expected_inbound_delay(flight_number: str, day: date) -> int:
    return int(_unit("delay", flight_number, day.isoformat()) ** 2 * 40)  # 0-40 min


def layover_risk(layover_min: int, airport, inbound_flight) -> dict:
    """Missed-connection risk = layover buffer + airport size + weather + delay history."""
    delay = expected_inbound_delay(inbound_flight.number, inbound_flight.arrive.date())
    eff = layover_min - delay
    buffer_pts = (55 if eff < 30 else 45 if eff < 45 else 32 if eff < 60
                  else 18 if eff < 90 else 8 if eff < 150 else 3)
    airport_pts = {3: 0, 2: 6, 1: 12}.get(airport.size, 6)  # fewer rebooking options at small airports
    sev, wx_label = simulated_weather(airport.code, inbound_flight.arrive.date())
    weather_pts = round(sev * 25)
    delay_pts = round(delay / 40 * 10)
    score = min(100, buffer_pts + airport_pts + weather_pts + delay_pts)
    level = "Low" if score < 25 else "Medium" if score < 45 else "High"
    return {
        "airport": airport.code, "minutes": layover_min, "score": score, "level": level,
        "weather": wx_label, "expected_delay": delay,
        "factors": {"layover buffer": buffer_pts, "airport size": airport_pts,
                    "weather": weather_pts, "delay history": delay_pts},
    }


# --------------------------------------------------------------------------
# Itineraries
# --------------------------------------------------------------------------
@dataclass
class Itinerary:
    legs: list
    price: int
    minutes: int
    layovers: list = field(default_factory=list)
    tags: list = field(default_factory=list)

    @property
    def stops(self) -> int:
        return len(self.legs) - 1

    @property
    def risk_score(self) -> int:
        return max((l["score"] for l in self.layovers), default=0)

    @property
    def risk_level(self) -> str:
        if not self.layovers:
            return "None"
        s = self.risk_score
        return "Low" if s < 25 else "Medium" if s < 45 else "High"

    @property
    def key(self) -> tuple:
        return tuple(f.id for f in self.legs)


def _build(path, price_of, airports) -> Itinerary:
    layovers = []
    for a, b in zip(path, path[1:]):
        gap = int((b.depart - a.arrive).total_seconds() // 60)
        layovers.append(layover_risk(gap, airports[a.dest], a))
    return Itinerary(
        legs=list(path),
        price=sum(price_of[f.id] for f in path),
        minutes=int((path[-1].arrive - path[0].depart).total_seconds() // 60),
        layovers=layovers,
    )


def _dijkstra(by_origin, origin, dest, day, objective, price_of, airports, max_legs, k):
    def edge(g, gap=0, risk=0):
        if objective == "price":
            return price_of[g.id]
        if objective == "duration":
            return gap + g.minutes
        return price_of[g.id] + RISK_PESO * risk

    counter = itertools.count()  # heap tie-breaker
    pq = []
    for f in by_origin[origin]:
        if f.depart.date() == day:
            heapq.heappush(pq, (edge(f), next(counter), f, (f,)))

    settled, results = set(), []
    while pq and len(results) < k:
        cost, _, f, path = heapq.heappop(pq)
        state = (f.id, len(path))
        if state in settled:
            continue
        settled.add(state)
        if f.dest == dest:
            results.append(path)
            continue
        if len(path) >= max_legs:
            continue
        seen = {p.origin for p in path}
        for g in by_origin[f.dest]:
            gap = int((g.depart - f.arrive).total_seconds() // 60)
            if gap < MIN_LAYOVER or gap > MAX_LAYOVER or g.dest in seen:
                continue
            risk = layover_risk(gap, airports[f.dest], f)["score"] if objective == "safe" else 0
            heapq.heappush(pq, (cost + edge(g, gap, risk), next(counter), g, path + (g,)))
    return results


def find_itineraries(flights, airports, origin, dest, day, price_of,
                     max_stops=1, per_objective=6) -> list[Itinerary]:
    """Return de-duplicated itineraries tagged Cheapest / Fastest / Safest."""
    by_origin = defaultdict(list)
    for f in flights:
        by_origin[f.origin].append(f)
    found: dict[tuple, Itinerary] = {}
    best_by: dict[str, Itinerary] = {}
    for objective, tag in (("price", "Cheapest"), ("duration", "Fastest"), ("safe", "Safest")):
        paths = _dijkstra(by_origin, origin, dest, day, objective, price_of,
                          airports, max_stops + 1, per_objective)
        for p in paths:
            it = _build(p, price_of, airports)
            found.setdefault(it.key, it)
        if paths:
            best_by[tag] = found[tuple(f.id for f in paths[0])]
    for tag, it in best_by.items():
        if tag not in it.tags:
            it.tags.append(tag)
    return list(found.values())
