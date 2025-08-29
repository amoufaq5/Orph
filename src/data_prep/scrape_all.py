import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "src" / "data_prep" / "scrapers"

SCRIPTS = [
    "medlineplus_fetch.py",
    "clinicaltrials_fetch.py",
    "openfda_labels_fetch.py",
    "pubmed_fetch.py",
]

def run(p):
    cmd = [sys.executable, str(SCR / p)]
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=False, cwd=str(ROOT))

def main():
    for s in SCRIPTS:
        run(s)

if __name__ == "__main__":
    main()
