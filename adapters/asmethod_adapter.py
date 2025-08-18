from orph_tools.logic.asmethod_parser import ASMethodParser

_parser = ASMethodParser()

def parse_asmethod(text: str) -> dict:
    """Adapter for ASMETHOD parsing"""
    return _parser.parse(text)
