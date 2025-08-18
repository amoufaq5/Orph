from orph_tools.models.fusion_engine import FusionEngine

_fusion = FusionEngine()

def fuse_results(inputs: dict) -> dict:
    """Adapter for fusion engine"""
    return _fusion.combine(inputs)
