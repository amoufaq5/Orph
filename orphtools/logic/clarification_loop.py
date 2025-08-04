# orphtools/logic/clarification_loop.py
class ClarificationLoop:
    def __init__(self, questions_per_turn=1, max_rounds=3):
        self.questions_per_turn = questions_per_turn
        self.max_rounds = max_rounds
        self.clarification_prompts = {
            "duration_days": "How long have you had this symptom? (in days)",
            "severity": "How severe is the symptom? (mild, moderate, severe)",
            "location": "Where exactly do you feel the symptom?",
            "onset": "When did the symptom start?",
            "relieving_factors": "What makes the symptom better or worse?"
        }

    def get_questions(self, missing_keys):
        return [self.clarification_prompts[k] for k in missing_keys if k in self.clarification_prompts][:self.questions_per_turn]

    def update_profile(self, user_profile, answers):
        for key, value in answers.items():
            user_profile[key] = value
        return user_profile


# Example usage:
# clarifier = ClarificationLoop()
# q = clarifier.get_questions(["duration_days", "severity"])
# profile = clarifier.update_profile({}, {"duration_days": 3, "severity": "moderate"})
