"""Utility helpers for model I/O and dirs."""
import os
import joblib


def ensure_dirs():
    os.makedirs("models/trained", exist_ok=True)
    os.makedirs("models/evaluation", exist_ok=True)
    os.makedirs("models/predictions", exist_ok=True)


def save_joblib(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str):
    return joblib.load(path)