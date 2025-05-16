# medline_scraper.py

import os
import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://medlineplus.gov"
TOPIC_URL = "https://medlineplus.gov/encyclopedia.html"


def scrape_medlineplus(output_path):
    results = []
    try:
        print("🌐 Fetching MedlinePlus index...")
        response = requests.get(TOPIC_URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select("ul.alpha-links li a")

        for link in links[:30]:  # Limit for MVP
            title = link.text.strip()
            url = BASE_URL + link.get("href")
            print(f"📘 Scraping: {title}")
            page = requests.get(url)
            page.raise_for_status()
            inner_soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = inner_soup.select("#ency_summary p")
            summary = " ".join(p.text.strip() for p in paragraphs)

            results.append({
                "title": title,
                "url": url,
                "summary": summary
            })

        output_file = os.path.join(output_path, "medlineplus_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved {len(results)} entries to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping MedlinePlus: {e}")
