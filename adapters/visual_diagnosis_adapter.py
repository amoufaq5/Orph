from orph_tools.models.visual_diagnosis import VisualDiagnosis

_visual = VisualDiagnosis()

def analyze_image(image_path: str) -> dict:
    """Adapter for visual diagnosis engine"""
    return _visual.analyze(image_path)
