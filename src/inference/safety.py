def disclaimers(role: str) -> str:
    if role == "patient":
        return ("⚠️ Not a medical diagnosis. If you have severe symptoms "
                "or feel unwell, seek urgent medical care.")
    return "Research-use only. Do not rely on this for clinical decisions."
