from orph_tools.logic.referral_generator import ReferralGenerator

_ref = ReferralGenerator()

def generate_referral(data: dict) -> dict:
    """Adapter between old referral generator and new API"""
    return _ref.run(data)
