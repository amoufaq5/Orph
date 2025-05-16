# webmd_scraper.py

import os
import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.webmd.com"
SYMPTOMS_URL = "https://www.webmd.com/a-to-z-guides/condition-az"

def scrape_webmd(output_path):
    symptoms_data = []
    try:
        resp = requests.get(SYMPTOMS_URL, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        condition_links = soup.select(".alphabetical-list li a")

        for link in condition_links[:30]:  # Limit for MVP
            title = link.text.strip()
            url = BASE_URL + link.get("href")
            print(f"🩺 Scraping {title}...")
            page = requests.get(url)
            inner_soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = inner_soup.select("article p")
            text = " ".join([p.text.strip() for p in paragraphs])
            symptoms_data.append({
                "disease": title,
                "url": url,
                "content": text
            })

        output_file = os.path.join(output_path, "webmd_conditions.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(symptoms_data, f, indent=2)
        print(f"✅ Saved {len(symptoms_data)} entries to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping WebMD: {e}")
