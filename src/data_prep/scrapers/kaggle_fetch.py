import os, sys, json, zipfile, argparse, subprocess, pathlib, time
from typing import Dict, List, Any

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

DEFAULT_OUT = "/workspace/data/raw/kaggle"
DEFAULT_CATALOG = "config/kaggle_catalog.yaml"

# --------------------- utils ---------------------
def log(*a): print("[kaggle]", *a)

def ensure_kaggle_creds_from_env():
    """Create ~/.kaggle/kaggle.json from env vars if not present."""
    kpath = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kpath):
        return
    user, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    if not user or not key:
        log("WARNING: no ~/.kaggle/kaggle.json and no KAGGLE_USERNAME/KAGGLE_KEY env vars.")
        log("         Kaggle CLI will fail to download protected datasets.")
        return
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(kpath, "w") as f:
        json.dump({"username": user, "key": key}, f)
    os.chmod(kpath, 0o600)
    log("wrote ~/.kaggle/kaggle.json from env")

def run(cmd: List[str]) -> int:
    print("[cmd]", " ".join(cmd))
    return subprocess.call(cmd)

def unzip_all(src_dir: pathlib.Path, dest_dir: pathlib.Path):
    for z in src_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as zh:
                zh.extractall(dest_dir)
            # keep zips for provenance; comment the next line to delete them
            # z.unlink()
            log(f"unzipped: {z.name}")
        except Exception as e:
            log(f"ERROR unzip {z}: {e}")

def load_catalog(path: str) -> Dict[str, Any]:
    if yaml is None:
        log("pyyaml not installed; using built-in minimal catalog")
        return {
            "nlp_core": {
                "enabled": True,
                "items": [
                    {"key": "pubmed_qa",       "slug": "junseongkim/pmqa-pubmed-question-answer-dataset", "type": "dataset", "enabled": True},
                    {"key": "pubmed_200k_rct", "slug": "zhangjuefei/pubmed-rct",                           "type": "dataset", "enabled": True},
                    {"key": "medmcqa",         "slug": "vineethakkinapalli/medmcqa",                       "type": "dataset", "enabled": True},
                    {"key": "drugscom_reviews","slug": "jessicali9530/drug-reviews",                       "type": "dataset", "enabled": True},
                    {"key": "webmd_reviews",   "slug": "praveengovi/webmd-reviews-dataset",                "type": "dataset", "enabled": True},
                ],
            }
        }
    p = pathlib.Path(path)
    if not p.exists():
        log(f"catalog not found at {path}; using built-in minimal catalog")
        return load_catalog("__NOFILE__")  # fallback minimal
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

# --------------------- core ---------------------
def download_item(slug: str, kind: str, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle CLI: datasets download / competitions download
    if kind == "dataset":
        rc = run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "-q"])
    elif kind == "competition":
        rc = run(["kaggle", "competitions", "download", "-c", slug, "-p", str(out_dir), "-q"])
    else:
        log(f"Unknown type for {slug}: {kind}")
        return

    if rc != 0:
        log(f"FAILED download for {slug}")
        return

    unzip_all(out_dir, out_dir)

def gather_items(catalog: Dict[str, Any], only: List[str], enable_heavy: bool) -> List[Dict[str, Any]]:
    items = []
    for cat_name, cat in catalog.items():
        cat_enabled = bool(cat.get("enabled", False))
        if not cat_enabled and not enable_heavy:
            # allow heavy categories to be toggled on by --enable-heavy
            continue
        for it in cat.get("items", []):
            # respect each item's local enabled flag unless override via --only or --enable-heavy
            if only and it["key"] not in only:
                continue
            if not it.get("enabled", False) and not enable_heavy and not only:
                continue
            items.append(it)
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Root output dir (default: {DEFAULT_OUT})")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG, help="YAML catalog path")
    ap.add_argument("--only", nargs="+", default=None, help="Keys to download exclusively (e.g. pubmed_qa medmcqa)")
    ap.add_argument("--enable-heavy", action="store_true", help="Also allow disabled/huge categories & items")
    ap.add_argument("--max-items", type=int, default=None, help="Stop after N items (safety)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between items to be polite")
    ap.add_argument("--list", action="store_true", help="List items from catalog and exit")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)

    if args.list:
        print(json.dumps(catalog, indent=2))
        return

    ensure_kaggle_creds_from_env()

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    items = gather_items(catalog, args.only, args.enable_heavy)
    if not items:
        log("No items selected. Use --list, --only, or --enable-heavy.")
        return

    count = 0
    for it in items:
        key, slug, kind = it["key"], it["slug"], it.get("type", "dataset")
        dest = out_root / key
        log(f"downloading [{key}] {slug} -> {dest}")
        try:
            download_item(slug, kind, dest)
        except Exception as e:
            log(f"ERROR {key}: {e}")
        count += 1
        if args.max_items and count >= args.max_items:
            log(f"Reached max-items={args.max_items}, stopping")
            break
        if args.sleep > 0:
            time.sleep(args.sleep)

    log("done")

if __name__ == "__main__":
    main()
