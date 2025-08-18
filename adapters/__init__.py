"""
Adapters export surface for the Orph backend.

This file centralizes all adapter imports so API routes and services can do:
    from src.adapters import (
        normalize_chat_request, make_chat_response, make_error,
        generate_referral, clarify, parse_asmethod,
        run_diagnosis, fuse_results, classify_symptoms, analyze_image,
    )
and stay stable even if some adapters are not implemented yet.
"""

# --- Optional/robust import helper -------------------------------------------------
def _safe_import(module: str, names: list[str]) -> dict:
    out = {}
    try:
        mod = __import__(module, fromlist=names)
        for n in names:
            out[n] = getattr(mod, n)
    except Exception:
        # Missing adapter or missing dependency: skip silently.
        # You can log here if you prefer.
        pass
    return out

# --- IO adapters (required) --------------------------------------------------------
from .io_adapter import (
    normalize_chat_request,
    normalize_clarify_request,
    normalize_referral_request,
    normalize_image_request,
    make_chat_response,
    make_clarify_response,
    make_referral_response,
    make_error,
)

# --- Logic / model adapters (optional, loaded safely) ------------------------------
_globals = {}

_globals.update(_safe_import("src.adapters.referral_adapter", ["generate_referral"]))
_globals.update(_safe_import("src.adapters.clarification_adapter", ["clarify"]))
_globals.update(_safe_import("src.adapters.asmethod_adapter", ["parse_asmethod"]))

_globals.update(_safe_import("src.adapters.diagnosis_adapter", ["run_diagnosis"]))
_globals.update(_safe_import("src.adapters.fusion_adapter", ["fuse_results"]))
_globals.update(_safe_import("src.adapters.symptom_classifier_adapter", ["classify_symptoms"]))
_globals.update(_safe_import("src.adapters.visual_diagnosis_adapter", ["analyze_image"]))

globals().update(_globals)

# --- Public API -------------------------------------------------------------------
__all__ = [
    # IO
    "normalize_chat_request",
    "normalize_clarify_request",
    "normalize_referral_request",
    "normalize_image_request",
    "make_chat_response",
    "make_clarify_response",
    "make_referral_response",
    "make_error",
    # Logic / models (optional; may be missing until you add files)
    "generate_referral",
    "clarify",
    "parse_asmethod",
    "run_diagnosis",
    "fuse_results",
    "classify_symptoms",
    "analyze_image",
]
