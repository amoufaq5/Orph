from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

# ---------- Public types (lightweight; no pydantic required)

class ChatIn(TypedDict, total=False):
    message: str
    asmethod: Dict[str, str]
    context: Dict[str, Any]

class ChatOut(TypedDict, total=False):
    answer: str
    confidence: float
    follow_up: Optional[str]
    samples: Optional[List[str]]
    referral: Optional[Dict[str, Any]]

class ClarifyIn(TypedDict, total=False):
    confidence: Union[int, float]
    asmethod: Dict[str, str]

class ClarifyOut(TypedDict, total=False):
    needed: bool
    question: Optional[str]

class ReferralIn(TypedDict, total=False):
    asmethod: Dict[str, str]
    prediction: Dict[str, Any]  # e.g. {"diagnosis": "...", "confidence": 0.73, ...}

class ReferralOut(TypedDict, total=False):
    refer: bool
    reason: Optional[str]
    tags: List[str]

class ImageIn(TypedDict, total=False):
    image_path: Optional[str]   # if you pass a path
    # or, if your route handles multipart, you can pass bytes via the route directly


# ---------- Normalizers (JSON -> internal typed dicts)

def _require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{ctx}: missing keys {missing}")

def normalize_chat_request(payload: Dict[str, Any]) -> ChatIn:
    _require_keys(payload, ["message"], "ChatRequest")
    message = str(payload.get("message", "")).strip()
    asmethod = payload.get("asmethod") or {}
    context = payload.get("context") or {}
    return ChatIn(message=message, asmethod=asmethod, context=context)

def normalize_clarify_request(payload: Dict[str, Any]) -> ClarifyIn:
    _require_keys(payload, ["confidence", "asmethod"], "ClarifyRequest")
    conf_raw = payload.get("confidence", 0)
    try:
        confidence = float(conf_raw)
    except Exception:
        raise ValueError("ClarifyRequest: 'confidence' must be a number")
    asmethod = payload.get("asmethod") or {}
    return ClarifyIn(confidence=confidence, asmethod=asmethod)

def normalize_referral_request(payload: Dict[str, Any]) -> ReferralIn:
    _require_keys(payload, ["asmethod", "prediction"], "ReferralRequest")
    asmethod = payload.get("asmethod") or {}
    prediction = payload.get("prediction") or {}
    return ReferralIn(asmethod=asmethod, prediction=prediction)

def normalize_image_request(payload: Dict[str, Any]) -> ImageIn:
    # If you accept JSON {image_path}, normalize here; if multipart, handle in route
    path = payload.get("image_path")
    return ImageIn(image_path=path)


# ---------- Builders (internal -> JSON responses)

def make_chat_response(
    answer: str,
    confidence: float,
    follow_up: Optional[str] = None,
    samples: Optional[List[str]] = None,
    referral: Optional[Dict[str, Any]] = None,
) -> ChatOut:
    return ChatOut(
        answer=answer,
        confidence=float(confidence),
        follow_up=follow_up,
        samples=samples,
        referral=referral,
    )

def make_clarify_response(needed: bool, question: Optional[str]) -> ClarifyOut:
    return ClarifyOut(needed=bool(needed), question=question)

def make_referral_response(refer: bool, reason: Optional[str], tags: Optional[List[str]] = None) -> ReferralOut:
    return ReferralOut(refer=bool(refer), reason=reason, tags=(tags or []))


# ---------- Error helper

def make_error(message: str, code: str = "bad_request", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    err = {"error": {"code": code, "message": message}}
    if details:
        err["error"]["details"] = details
    return err
