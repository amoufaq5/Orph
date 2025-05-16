# orph_scraper_main.py

"""
Main Scraper Entry Point for Project Orph
This script includes modular scrapers for:
- WebMD
- Drugs.com
- OpenFDA
- PubMed
- MedlinePlus
- ClinicalTrials.gov

Each scraper will output JSON or CSV files to /data/raw
"""

import os
from scrapers.webmd_scraper import scrape_webmd
from scrapers.drugs_scraper import scrape_drugs
from scrapers.pubmed_scraper import scrape_pubmed
from scrapers.openfda_scraper import scrape_openfda
from scrapers.medline_scraper import scrape_medlineplus
from scrapers.clinical_trials_scraper import scrape_clinical_trials

RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)

def main():
    print("🔍 Starting Orph data scraping pipeline...")

    print("📦 Scraping WebMD...")
    scrape_webmd(RAW_DATA_PATH)

    print("💊 Scraping Drugs.com...")
    scrape_drugs(RAW_DATA_PATH)

    print("🧬 Scraping PubMed abstracts...")
    scrape_pubmed(RAW_DATA_PATH)

    print("📁 Scraping OpenFDA drug labels...")
    scrape_openfda(RAW_DATA_PATH)

    print("📚 Scraping MedlinePlus topics...")
    scrape_medlineplus(RAW_DATA_PATH)

    print("📝 Scraping ClinicalTrials.gov data...")
    scrape_clinical_trials(RAW_DATA_PATH)

    print("✅ All scraping complete. Raw data saved to /data/raw")

if __name__ == "__main__":
    main()
