"""ML fare predictor: "Buy Now" vs "Wait" with a confidence score.

Target: will this fare drop by >=5% within the next 7 days?
Model : RandomForestClassifier (scikit-learn).

IMPORTANT FOR YOUR DEFENSE: no real historical airfare dataset ships with this
project, so the model trains on a SYNTHETIC dataset generated from plausible
airline-pricing behaviour (far-out + empty flights tend to drop; close-in +
full flights tend to rise). To make it real, replace `synthetic_dataset()`
with rows loaded from a historical fare CSV that has the same FEATURES.
"""
import os

import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")
FEATURES = ["days_out", "load", "dow", "month", "route_demand", "base_price"]

_model = None
metrics = {"accuracy": None, "samples": 0, "source": "synthetic"}


def _logit(days_out, load, dow, route_demand):
    return (0.045 * (days_out - 20) - 3.4 * (load - 0.45)
            - 0.30 * (route_demand - 4) + 0.35 * (dow >= 4))


def synthetic_dataset(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.integers(0, 91, n),            # days_out
        rng.random(n),                     # load factor
        rng.integers(0, 7, n),             # day of week
        rng.integers(1, 13, n),            # month
        rng.integers(2, 7, n),             # route demand (sum of airport sizes)
        rng.integers(1500, 9000, n),       # base fare
    ]).astype(float)
    p = 1 / (1 + np.exp(-_logit(X[:, 0], X[:, 1], X[:, 2], X[:, 4])))
    y = (rng.random(n) < p).astype(int)
    return X, y


def get_model():
    global _model
    if _model is not None or not HAVE_SKLEARN:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception:
            pass
    X, y = synthetic_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    _model = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=1, n_jobs=1)
    _model.fit(Xtr, ytr)
    metrics.update(accuracy=round(float(accuracy_score(yte, _model.predict(Xte))), 3), samples=len(X))
    try:
        joblib.dump(_model, MODEL_PATH)
    except Exception:
        pass
    return _model


def predict(days_out: int, load: float, depart, route_demand: int, base_price: int) -> dict:
    x = np.array([[min(days_out, 90), load, depart.weekday(), depart.month, route_demand, base_price]], float)
    model = get_model()
    if model is not None:
        p_drop = float(model.predict_proba(x)[0][list(model.classes_).index(1)])
    else:  # graceful fallback if scikit-learn is missing
        p_drop = float(1 / (1 + np.exp(-_logit(x[0, 0], x[0, 1], x[0, 2], x[0, 4]))))
    wait = p_drop >= 0.5
    return {
        "advice": "Wait" if wait else "Buy Now",
        "confidence": round(max(p_drop, 1 - p_drop) * 100),
        "p_drop": round(p_drop, 3),
    }
