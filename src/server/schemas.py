from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Chat
@dataclass
class ChatRequest:
    message: str
    asmethod: Optional[Dict[str, str]] = None  # structured answers
    context: Optional[Dict[str, Any]] = None

@dataclass
class ChatResponse:
    answer: str
    confidence: float
    follow_up: Optional[str] = None
    samples: Optional[List[str]] = None

# Clarification
@dataclass
class ClarifyRequest:
    confidence: float
    asmethod: Dict[str, str]

@dataclass
class ClarifyResponse:
    needed: bool
    question: Optional[str]

# Referral
@dataclass
class ReferralRequest:
    asmethod: Dict[str, str]
    prediction: Dict[str, Any]

@dataclass
class ReferralResponse:
    refer: bool
    reason: Optional[str]
    tags: List[str]

# ASMETHOD parse
@dataclass
class ParseASMethodRequest:
    text: str

@dataclass
class ParseASMethodResponse:
    filled: Dict[str, str]
    missing: List[str]
