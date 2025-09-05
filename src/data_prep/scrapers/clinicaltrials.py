from typing import Iterator, Dict
from .base import Scraper, mk_id

class ClinicalTrialsScraper(Scraper):
    name = "clinicaltrials"
    def stream(self) -> Iterator[Dict]:
        # TODO: call v2 API; store inclusion/exclusion, outcomes, arms
        trial = {"nct":"NCT00000000","title":"Trial of X in Y","results":"X improved Y by 15%"}
        yield {
          "id": mk_id("ct"),
          "modality": ["text"],
          "task": "summarize",
          "text": f"{trial['title']} ({trial['nct']}): {trial['results']}",
          "answer": None,
          "rationale": None,
          "labels": {},
          "meta": {"source":"clinicaltrials","license":"public-domain"},
          "split":"train"
        }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ClinicalTrialsScraper(ap.parse_args().out).run()
