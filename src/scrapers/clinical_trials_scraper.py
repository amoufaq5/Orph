# clinical_trials_scraper.py

import os
import json
import requests

BASE_URL = "https://clinicaltrials.gov/api/query/full_studies"
PARAMS = {
    "expr": "covid OR cancer OR diabetes",
    "min_rnk": 1,
    "max_rnk": 10,
    "fmt": "json"
}

def scrape_clinical_trials(output_path):
    try:
        print("🔬 Fetching ClinicalTrials.gov data...")
        response = requests.get(BASE_URL, params=PARAMS, timeout=10)
        response.raise_for_status()
        data = response.json()
        studies = data.get("FullStudiesResponse", {}).get("FullStudies", [])

        trials = []
        for study in studies:
            protocol = study.get("Study", {}).get("ProtocolSection", {})
            id_info = protocol.get("IdentificationModule", {})
            status_info = protocol.get("StatusModule", {})
            conditions = protocol.get("ConditionsModule", {}).get("ConditionList", {}).get("Condition", [])

            trials.append({
                "nct_id": id_info.get("NCTId"),
                "title": id_info.get("BriefTitle"),
                "status": status_info.get("OverallStatus"),
                "conditions": conditions
            })

        output_file = os.path.join(output_path, "clinical_trials.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2)
        print(f"✅ Saved {len(trials)} clinical trials to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping ClinicalTrials.gov: {e}")
