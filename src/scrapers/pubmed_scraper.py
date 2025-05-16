# pubmed_scraper.py

import os
import json
from Bio import Entrez

# Set your email for NCBI API usage
Entrez.email = "your_email@example.com"

# Example disease queries
QUERIES = ["influenza", "pneumonia", "diabetes", "covid", "hepatitis"]


def scrape_pubmed(output_path):
    abstracts_data = []
    try:
        for query in QUERIES:
            print(f"🔎 Searching PubMed for: {query}")
            handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
            record = Entrez.read(handle)
            id_list = record["IdList"]

            if not id_list:
                continue

            handle = Entrez.efetch(db="pubmed", id=','.join(id_list), rettype="abstract", retmode="text")
            raw_data = handle.read()

            for idx, abs_text in zip(id_list, raw_data.split("\n\n")):
                abstracts_data.append({
                    "query": query,
                    "pubmed_id": idx,
                    "abstract": abs_text.strip()
                })

        output_file = os.path.join(output_path, "pubmed_abstracts.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(abstracts_data, f, indent=2)
        print(f"✅ Saved {len(abstracts_data)} abstracts to {output_file}")

    except Exception as e:
        print(f"❌ Error scraping PubMed: {e}")
