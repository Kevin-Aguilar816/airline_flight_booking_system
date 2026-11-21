"""Shared test setup: one temporary database for the whole test session.
Environment variables must be set BEFORE `app` is imported anywhere."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("MODEL_PATH", f"{_tmp}/m.joblib")
