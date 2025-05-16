# openfda_scraper.py

import os
import json
import requests

OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
PARAMS = {
    "limit": 100,
    "search": "active_ingredient:acetaminophen+OR+active_ingredient:ibuprofen"
}

def scrape_openfda(output_path):
    try:
        print("🔗 Connecting to OpenFDA API...")
        response = requests.get(OPENFDA_ENDPOINT, params=PARAMS, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        entries = []
        for entry in results:
            entries.append({
                "id": entry.get("id"),
                "brand_name": entry.get("openfda", {}).get("brand_name", [None])[0],
                "generic_name": entry.get("openfda", {}).get("generic_name", [None])[0],
                "purpose": entry.get("purpose", [""])[0],
                "warnings": entry.get("warnings", [""])[0],
                "dosage": entry.get("dosage_and_administration", [""])[0]
            })

        output_file = os.path.join(output_path, "openfda_labels.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"✅ Saved {len(entries)} entries to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping OpenFDA: {e}")
