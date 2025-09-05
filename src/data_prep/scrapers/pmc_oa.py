from typing import Iterator, Dict
from .base import Scraper, mk_id

class PMCOAScraper(Scraper):
    name = "pmc_oa"
    def stream(self) -> Iterator[Dict]:
        # TODO: parse OA XML to text; store license from article metadata
        yield {
          "id": mk_id("pmc"),
          "modality": ["text"],
          "task": "qa",
          "text": "Q: What is the first-line treatment for H. pylori?\nContext: ...",
          "answer": "Triple therapy (PPI + clarithromycin + amoxicillin) where resistance is low.",
          "rationale": "Guidelines recommend ...",
          "labels": {},
          "meta": {"source":"pmc_oa","license":"CC-BY"},
          "split":"train"
        }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    PMCOAScraper(ap.parse_args().out).run()
