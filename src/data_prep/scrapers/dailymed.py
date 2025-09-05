from typing import Iterator, Dict
from .base import Scraper, mk_id

class DailyMedScraper(Scraper):
    name = "dailymed"
    def stream(self) -> Iterator[Dict]:
        # TODO: fetch SPL JSON / XML; extract indications, contraindications, interactions
        sample = {
          "drug":"ibuprofen","indications":"pain, fever","contra":"GI bleed","ddi":"warfarin (bleeding risk)"
        }
        yield {
          "id": mk_id("dailymed"),
          "modality": ["text"],
          "task": "ddi",
          "text": f"{sample['drug']} label: indications={sample['indications']}; contraindications={sample['contra']}; interactions={sample['ddi']}",
          "answer": None,
          "rationale": None,
          "labels": {},
          "meta": {"source":"dailymed","license":"public-domain"},
          "split":"train"
        }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    DailyMedScraper(ap.parse_args().out).run()
