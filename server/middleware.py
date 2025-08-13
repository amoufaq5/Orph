from __future__ import annotations
from functools import wraps
from flask import request, jsonify
import os

API_KEY_ENV = "ORPH_API_KEY"


def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != os.getenv(API_KEY_ENV, "devkey"):
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper
