from orph_tools.logic.clarification_loop import ClarificationLoop

_clarifier = ClarificationLoop()

def clarify(data: dict) -> dict:
    """Adapter for clarification loop"""
    return _clarifier.run(data)
