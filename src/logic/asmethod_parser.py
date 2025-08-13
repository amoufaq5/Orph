# orphtools/logic/asmethod_parser.py
import numpy as np

class ASMethodParser:
    def __init__(self):
        self.feature_map = [
            "age", "is_self", "medications", "extra_meds", "duration_days",
            "history", "other_symptoms", "danger_symptoms"
        ]

    def parse(self, intake_dict):
        vector = []
        for key in self.feature_map:
            val = intake_dict.get(key, None)
            if key == "age":
                vector.append(val / 100 if val else 0.0)
            elif isinstance(val, (int, float)):
                vector.append(val)
            elif isinstance(val, str):
                vector.append(1.0 if val.strip().lower() not in ["", "none", "no"] else 0.0)
            elif isinstance(val, list):
                vector.append(len(val) / 10.0)
            else:
                vector.append(0.0)
        return np.array(vector, dtype=np.float32)


# Example usage:
# parser = ASMethodParser()
# vector = parser.parse({"age": 30, "danger_symptoms": ["bleeding"]})
