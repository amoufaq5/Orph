from __future__ import annotations
from functools import wraps
from flask import request, jsonify
import os

API_KEY_ENV = "ORPH_API_KEY"


def require_api_key(fn):
    """Protect endpoints with X-API-Key header.
    Usage:
        @app.post("/api/chat")
        @require_api_key
        def chat():
            ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        sent = request.headers.get("X-API-Key")
        want = os.getenv(API_KEY_ENV, "devkey")
        if not sent or sent != want:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper
