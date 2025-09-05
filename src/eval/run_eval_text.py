import os, json, time, argparse, random
from typing import Dict
from src.eval.datasets import load_pubmedqa, load_mcq_csv, load_shortqa
from src.eval.metrics import exact_match, f1, mcq_acc
from src.inference.llm import OrphLLM
from src.inference.pipelines import Pipeline
from src.utils.config import load_config
from src.utils.io import ensure_dir

def prompt_mcq(q: str, options: list[str]) -> str:
    letters = ["A","B","C","D"]
    opts = "\n".join([f"{letters[i]}) {options[i]}" for i in range(len(options))])
    return (
        "You are a concise medical expert. Choose the single best answer (A/B/C/D). "
        "Return only the letter.\n\n"
        f"Question: {q}\n{opts}\nAnswer:"
    )

def prompt_short(q: str, ctx: str) -> str:
    ctx_part = f"Context: {ctx}\n" if ctx else ""
    return (
        "Answer the medical question briefly and factually.\n"
        f"{ctx_part}Question: {q}\nAnswer:"
    )

def prompt_pubmedqa(q: str, ctx: str) -> str:
    ctx_part = f"Context: {ctx}\n" if ctx else ""
    return (
        "Answer 'yes' or 'no' based on evidence.\n"
        f"{ctx_part}Question: {q}\nAnswer:"
    )

def run(dataset: str, path: str, mode: str, role: str, limit: int | None, seed: int, out_path: str):
    random.seed(seed)
    cfg = load_config()
    ensure_dir(os.path.dirname(out_path))

    use_rag = (mode.lower() == "rag")
    if use_rag:
        pipe = Pipeline(index_dir=cfg.main.get("paths",{}).get("rag_index","data/artifacts/rag"),
                        top_k=cfg.main.get("rag",{}).get("top_k",5))
        llm = None
    else:
        llm = OrphLLM(cfg.main.get("inference",{}).get("model_dir","./out/text_orphgpt"),
                      device=cfg.main.get("inference",{}).get("device","auto"))
        pipe = None

    # load data
    if dataset == "pubmedqa":
        rows = list(load_pubmedqa(path))
    elif dataset in ("medmcqa","medqa"):
        rows = list(load_mcq_csv(path))
    elif dataset == "shortqa":
        rows = list(load_shortqa(path))
    else:
        raise ValueError("dataset must be one of: pubmedqa | medmcqa | medqa | shortqa")

    if limit:
        rows = rows[:limit]

    scores = []
    start = time.time()

    for i, r in enumerate(rows, 1):
        if r["type"] == "mcq":
            if use_rag:
                q = prompt_mcq(r["question"], r["options"])
                ans = pipe.answer(role, q)["answer"]
            else:
                ans = llm.generate(prompt_mcq(r["question"], r["options"]), max_new_tokens=8, temperature=0.0)
            # Extract letter
            letter = (ans.strip().lower() + " ")[0]
            pred_idx = {"a":0,"b":1,"c":2,"d":3}.get(letter, -1)
            sc = mcq_acc(pred_idx, r["label"])
            scores.append({"id":i, "metric":"acc", "score":sc})

        elif r["type"] == "yn":
            if use_rag:
                q = prompt_pubmedqa(r["question"], r.get("context",""))
                ans = pipe.answer(role, q)["answer"]
            else:
                ans = llm.generate(prompt_pubmedqa(r["question"], r.get("context","")), max_new_tokens=8, temperature=0.0)
            pred = "yes" if "yes" in ans.lower()[:10] else ("no" if "no" in ans.lower()[:10] else "unknown")
            gold = r["answer"] if r["answer"] in ("yes","no") else "unknown"
            sc = 1.0 if pred == gold else 0.0
            scores.append({"id":i, "metric":"acc", "score":sc})

        else:  # short answer
            if use_rag:
                q = prompt_short(r["question"], r.get("context",""))
                ans = pipe.answer(role, q)["answer"]
            else:
                ans = llm.generate(prompt_short(r["question"], r.get("context","")), max_new_tokens=64, temperature=0.2)
            scores.append({"id":i, "metric":"f1", "score":f1(ans, r["answer"])})
            scores.append({"id":i, "metric":"em", "score":exact_match(ans, r["answer"])})

        if i % 20 == 0:
            elapsed = time.time() - start
            print(f"[{i}/{len(rows)}] elapsed={elapsed:.1f}s")

    # aggregate
    agg: Dict[str, float] = {}
    for m in set(s["metric"] for s in scores):
        vals = [x["score"] for x in scores if x["metric"]==m]
        agg[m] = sum(vals)/max(1,len(vals))

    result = {
        "dataset": dataset,
        "mode": mode,
        "role": role,
        "n": len(rows),
        "metrics": agg,
        "timestamp": int(time.time())
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")
    print("Metrics:", agg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pubmedqa","medmcqa","medqa","shortqa"])
    ap.add_argument("--path", required=True, help="Path to dataset file (JSONL or CSV depending on dataset)")
    ap.add_argument("--mode", default="rag", choices=["rag","llm"])
    ap.add_argument("--role", default="clinician", choices=["patient","clinician","pharma","student"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/artifacts/eval/result.json")
    args = ap.parse_args()
    run(args.dataset, args.path, args.mode, args.role, args.limit, args.seed, args.out)
