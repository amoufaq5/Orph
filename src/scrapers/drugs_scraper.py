# drugs_scraper.py

import os
import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.drugs.com"
A_Z_URL = "https://www.drugs.com/drug_information.html"


def scrape_drugs(output_path):
    results = []
    try:
        print("📑 Fetching drug A-Z index...")
        response = requests.get(A_Z_URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        drug_links = soup.select("ul.ddc-list-column-2 a")

        for link in drug_links[:30]:  # Limit to 30 for MVP
            drug_name = link.text.strip()
            drug_url = BASE_URL + link.get("href")
            print(f"💊 Scraping {drug_name}...")
            
            drug_page = requests.get(drug_url)
            inner_soup = BeautifulSoup(drug_page.text, "html.parser")
            
            # Extract drug content
            summary = " ".join(p.text.strip() for p in inner_soup.select("#content p")[:5])

            results.append({
                "drug": drug_name,
                "url": drug_url,
                "summary": summary
            })

        # Save to file
        output_file = os.path.join(output_path, "drugs_com_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved {len(results)} entries to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping Drugs.com: {e}")
