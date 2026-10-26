"""Compare a new BO run (re-run) with a saved original run, campaign by campaign (matched by folder path)."""
import sys, glob, os, re
import numpy as np, pandas as pd

pd.set_option("display.width", 220)
new, old, label = sys.argv[1], sys.argv[2], sys.argv[3]


def norm(rel):
    parts = [p for p in re.split(r"[\\/]+", rel) if p not in (".", "")]
    parts = [p for p in parts if not p.startswith("BO_Cont")]
    return "/".join(parts)


def collect(run, name):
    out = {}
    for f in glob.glob(os.path.join(run, "**", name), recursive=True):
        out[norm(os.path.relpath(f, run))] = pd.read_csv(f)
    return out


sa, sb = collect(new, "bo_best_summary.csv"), collect(old, "bo_best_summary.csv")
keys = sorted(set(sa) & set(sb))
print(f"#### {label}: new {len(sa)} summaries, old {len(sb)}, matched by path {len(keys)}")
rows = []
for k in keys:
    x, y = sa[k].iloc[0], sb[k].iloc[0]
    parts = k.split("/")
    fam_x = x["best_family"] if "best_family" in x.index else x["family_fixed"]
    fam_y = y["best_family"] if "best_family" in y.index else y["family_fixed"]
    rows.append(dict(path=k, obj=x["objective"], mode=x["mode"], fam=fam_x,
                     new=float(x["Best_Objective"]), old=float(y["Best_Objective"]),
                     same_design=all(np.isclose(float(x[c]), float(y[c])) for c in ["UCx_used (mm)", "UCy_used (mm)", "UCz_used (mm)", "t_used (mm)"]),
                     same_family=(fam_x == fam_y)))
df = pd.DataFrame(rows)
df["absdiff"] = (df["new"] - df["old"]).abs()
df["reldiff"] = df["absdiff"] / df["old"].abs()
print("Best_Objective: identical(<1e-9) %d / %d ; within 1e-3 rel %d ; max abs diff %.4g ; max rel diff %.4g" %
      ((df.absdiff < 1e-9).sum(), len(df), (df.reldiff < 1e-3).sum(), df.absdiff.max(), df.reldiff.max()))
print("identical best design: %d / %d" % (df.same_design.sum(), len(df)))
for (o, mo), s in df.groupby(["obj", "mode"]):
    print(f"  {o:2s} {mo:9s} n={len(s):3d} identical={(s.absdiff<1e-9).sum():3d} same_design={s.same_design.sum():3d} max_rel={s.reldiff.max():.3g}")

la, lb = collect(new, "bo_log.csv"), collect(old, "bo_log.csv")
common = sorted(set(la) & set(lb))
nid, nfirst = 0, 0
for k in common:
    x, y = la[k], lb[k]
    if len(x) == len(y) and np.allclose(x["Objective_value"].values, y["Objective_value"].values, rtol=1e-9, atol=1e-9):
        nid += 1
    if len(x) >= 10 and len(y) >= 10 and np.allclose(x["Objective_value"].values[:10], y["Objective_value"].values[:10], rtol=1e-9):
        nfirst += 1
print(f"bo_log: first 10 (LHS) evaluations identical in {nfirst} of {len(common)} campaigns; all 100 iterations identical in {nid}")

# aggregate view used in the paper: best over seeds
for obj in ("YM", "SS"):
    for mo in ("Mixed",):
        s = df[(df.obj == obj) & (df["mode"] == mo)]
        print(f"best over seeds, {obj} {mo}: new {s.new.max():.5f}  old {s.old.max():.5f}")
df.to_csv(os.path.join(new, "compare_vs_original.csv"), index=False)
