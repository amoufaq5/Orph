import os, json, glob, time
from tabulate import tabulate
from src.utils.io import ensure_dir

def load_results(pattern="data/artifacts/eval/*.json"):
    rows = []
    for p in glob.glob(pattern):
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows.append(j)
    return rows

def build_tables(results):
    headers = ["Dataset","Mode","Role","N","Acc","F1","EM","Timestamp"]
    table = []
    for r in results:
        acc = r["metrics"].get("acc","-")
        f1  = r["metrics"].get("f1","-")
        em  = r["metrics"].get("em","-")
        ts  = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("timestamp",0)))
        table.append([r["dataset"], r["mode"], r["role"], r["n"], 
                      f"{acc:.3f}" if isinstance(acc,float) else acc,
                      f"{f1:.3f}" if isinstance(f1,float) else f1,
                      f"{em:.3f}" if isinstance(em,float) else em,
                      ts])
    return headers, table

def write_markdown(results, md_path="docs/LEADERBOARD.md"):
    ensure_dir(os.path.dirname(md_path))
    headers, table = build_tables(results)
    md = "# Orph Research — Evaluation Leaderboard\n\n"
    md += tabulate(table, headers=headers, tablefmt="github")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {md_path}")

def write_html(results, html_path="docs/leaderboard.html"):
    ensure_dir(os.path.dirname(html_path))
    headers, table = build_tables(results)
    # Minimal static HTML
    rows = "\n".join([
        "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"
        for row in table
    ])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Orph Leaderboard</title>
<style>body{{font-family:system-ui;margin:24px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:8px}}</style>
</head><body>
<h1>Orph Research — Evaluation Leaderboard</h1>
<table><thead><tr>{"".join([f"<th>{h}</th>" for h in headers])}</tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {html_path}")

if __name__ == "__main__":
    res = load_results("data/artifacts/eval/*.json")
    write_markdown(res)
    write_html(res)
