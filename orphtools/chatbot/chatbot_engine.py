# orphtools/chatbot/chatbot_engine.py
class ChatbotEngine:
    def __init__(self, diagnosis_engine, clarification_loop, referral_generator, threshold=0.75):
        self.diagnosis_engine = diagnosis_engine
        self.clarification_loop = clarification_loop
        self.referral_generator = referral_generator
        self.threshold = threshold
        self.user_profile = {}

    def process_input(self, user_input):
        print(f"👤 User: {user_input}")

        if "symptom" not in self.user_profile:
            self.user_profile["symptom"] = user_input
        
        results = self.diagnosis_engine.predict(user_input)

        if not results:
            print("🤖 I'm not confident enough to suggest a diagnosis yet. Let's clarify.")
            questions = self.clarification_loop.get_questions(["duration_days", "severity"])
            return questions

        print("🤖 Possible conditions:")
        for disease_idx, score in results:
            print(f"- Disease {disease_idx} (confidence: {score:.2f})")

        if any(score > 0.9 for _, score in results):
            referral_reason = "High confidence in severe symptoms. Referral recommended."
            self.referral_generator.generate_referral_pdf(self.user_profile, [user_input], referral_reason)
            self.referral_generator.log_referral(self.user_profile, [user_input], referral_reason)

        return results


# Example usage:
# chatbot = ChatbotEngine(diagnosis_engine, clarifier, referral_generator)
# chatbot.process_input("I feel chest pain and dizziness")
