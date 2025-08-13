from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ChatRequest:
    message: str
    context: Optional[Dict] = None

@dataclass
class ChatResponse:
    answer: str
    confidence: float
    rationale: str | None = None
    follow_up: str | None = None
