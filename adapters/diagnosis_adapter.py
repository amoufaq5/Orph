from orph_tools.models.diagnosis_engine import DiagnosisEngine

_diag = DiagnosisEngine()

def run_diagnosis(data: dict) -> dict:
    """Adapter for diagnosis engine"""
    return _diag.predict(data)
