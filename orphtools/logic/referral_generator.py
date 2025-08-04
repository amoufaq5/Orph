# orphtools/logic/referral_generator.py
import os
import json
from datetime import datetime
from fpdf import FPDF

class ReferralGenerator:
    def __init__(self, referral_dir="referrals/pdf/", log_dir="logs/sessions/"):
        self.referral_dir = referral_dir
        self.log_dir = log_dir
        os.makedirs(self.referral_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def generate_referral_pdf(self, patient_info, symptoms, reason):
        filename = f"referral_{patient_info.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.referral_dir, filename)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Medical Referral", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"Patient ID: {patient_info.get('id', 'N/A')}\n"
                                   f"Age: {patient_info.get('age', 'N/A')}\n"
                                   f"Symptoms: {', '.join(symptoms)}\n"
                                   f"Reason for Referral: {reason}")
        pdf.output(filepath)
        print(f"✅ Referral PDF saved to {filepath}")
        return filepath

    def log_referral(self, patient_info, symptoms, reason):
        log_filename = f"referral_{datetime.now().strftime('%Y%m%d')}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        record = {
            "timestamp": datetime.now().isoformat(),
            "patient": patient_info,
            "symptoms": symptoms,
            "reason": reason
        }

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(record)

        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"✅ Referral logged to {log_path}")
        return log_path


# Example usage:
# generator = ReferralGenerator()
# pdf_path = generator.generate_referral_pdf({"id": "U123", "age": 30}, ["fever", "cough"], "Severe symptoms")
# log_path = generator.log_referral({"id": "U123", "age": 30}, ["fever", "cough"], "Severe symptoms")
