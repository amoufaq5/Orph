import glob, os, json
from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logger import get_logger
from src.data_prep.cleaners.text_clean import normalize_text
from src.data_prep.labeling import umls_map

log = get_logger("merge")

def unify_row(r):
    # Ensure schema v2 required keys exist
    r.setdefault("id", None)
    r.setdefault("modality", ["text"])
    r.setdefault("task", "summarize")
    r.setdefault("text", None)
    r.setdefault("image_path", None)
    r.setdefault("answer", None)
    r.setdefault("rationale", None)
    r.setdefault("labels", {})
    r.setdefault("meta", {"source": "unknown", "license": "unknown"})
    r.setdefault("split", "train")
    if r["text"]:
        r["text"] = normalize_text(r["text"])
    # Optional: auto-label map
    if r["text"]:
        r["labels"].setdefault("icd10", umls_map.map_icd10(r["text"]))
        r["labels"].setdefault("snomed", umls_map.map_snomed(r["text"]))
    return r

def merge_dirs(input_dirs, out_path):
    ensure_dir(os.path.dirname(out_path))
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for d in input_dirs:
            for p in glob.glob(os.path.join(d, "*.jsonl")):
                for r in read_jsonl(p):
                    u = unify_row(r)
                    out_f.write(json.dumps(u, ensure_ascii=False) + "\n")
                    count += 1
    log.info(f"Merged {count} rows → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge_dirs(args.inputs, args.out)
