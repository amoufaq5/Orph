from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

ASMETHOD_ORDER = [
    ("age", "What is the patient's age and general appearance?"),
    ("self_or_someone", "Is this for you or someone else?"),
    ("medication", "Are you currently taking any medications?"),
    ("extra_meds", "Any supplements or over-the-counter drugs?"),
    ("time", "When did the symptoms start and how have they changed?"),
    ("history", "Any relevant medical history or chronic conditions?"),
    ("other_symptoms", "Any other symptoms you’ve noticed?"),
    ("danger", "Any danger signs like severe chest pain, breathing difficulty, loss of consciousness, high fever in infants, or uncontrolled bleeding?")
]

@dataclass
class ASMethodState:
    answers: Dict[str, str]

    def unanswered(self) -> List[str]:
        return [k for k, _ in ASMETHOD_ORDER if k not in self.answers or not self.answers[k]]

    def next_question(self) -> str | None:
        for key, q in ASMETHOD_ORDER:
            if key not in self.answers or not self.answers[key]:
                return q
        return None
