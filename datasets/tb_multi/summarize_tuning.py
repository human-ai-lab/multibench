"""Apply the pre-declared selection rule of results/tuning_protocol.md to results/loco_tune (seed 0, 4 public LOCO folds)."""
import glob, json, re, collections
import numpy as np
R = collections.defaultdict(dict)
for f in glob.glob("results/loco_tune/tune_*_s0.json"):
    m = re.match(r".*/tune_(.+)_(qatar|nlm|tbx11k|pakistan)_s0\.json", f)
    R[m.group(1)][m.group(2)] = json.load(open(f))["auc"]
def parse(c):  # -> backbone, lr, unfreeze depth
    b, lr, u = re.match(r"(vit|raddino|resnet50|xrv)_lr([\de.-]+)_[uc](\d+)", c).groups()
    return b, float(lr), int(u)
out = ["| backbone | config | qatar | nlm | tbx11k | pakistan | mean(4) | mean(nlm,tbx11k) |", "|---|---|---|---|---|---|---|---|"]
sel = {}
for b in ["vit", "raddino", "resnet50", "xrv"]:
    rows = sorted([(c, R[c]) for c in R if parse(c)[0] == b], key=lambda x: (parse(x[0])[2], parse(x[0])[1]))
    for c, r in rows:
        out.append(f"| {b} | {c} | " + " | ".join(f"{r[t]:.3f}" for t in ["qatar", "nlm", "tbx11k", "pakistan"]) +
                   f" | {np.mean(list(r.values())):.3f} | {(r['nlm'] + r['tbx11k']) / 2:.3f} |")
    for name, key in (("mean4", lambda r: np.mean(list(r.values()))), ("nlm+tbx", lambda r: (r["nlm"] + r["tbx11k"]) / 2)):
        best = max(key(r) for _, r in rows)
        tied = [c for c, r in rows if key(r) >= best - 0.01]
        pick = min(tied, key=lambda c: (parse(c)[2], parse(c)[1]))  # fewer unfrozen blocks, then lower lr
        sel[(b, name)] = (pick, key(dict(rows)[pick]), len(tied))
out.append("\n| backbone | rule | selected | score | configs tied within 0.01 |\n|---|---|---|---|---|")
for (b, n), (c, v, k) in sel.items():
    out.append(f"| {b} | {n} | {c} | {v:.3f} | {k} |")
open("results/tuning_summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
