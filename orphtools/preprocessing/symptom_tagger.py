# orphtools/preprocessing/symptom_tagger.py
import spacy
import re

class SymptomTagger:
    def __init__(self, symptom_keywords=None):
        self.symptom_keywords = symptom_keywords or [
            "fever", "cough", "pain", "nausea", "vomiting", "rash", "headache",
            "fatigue", "diarrhea", "shortness of breath", "dizziness"
        ]
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            raise RuntimeError("SpaCy model not found. Run: python -m spacy download en_core_web_sm")

    def tag_text(self, text):
        if not isinstance(text, str):
            return []
        doc = self.nlp(text.lower())
        matches = [token.text for token in doc if token.text in self.symptom_keywords]
        return list(set(matches))

    def tag_dataframe(self, df, text_column, output_column="symptoms_tagged"):
        df[output_column] = df[text_column].fillna("").apply(self.tag_text)
        return df


# Example usage:
# from orphtools.preprocessing.symptom_tagger import SymptomTagger
# tagger = SymptomTagger()
# tagged_df = tagger.tag_dataframe(df, text_column="overview")
