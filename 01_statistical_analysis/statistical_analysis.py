"""Statistical analysis of the 270-design TPMS FE dataset.

Reproduces Section 3.2 numbers (overall / per-family statistics) and
Supplementary Figs. S1-S4 (histograms and box plots of E and E_specific).

Original notebook: dataset_analysis_statistical 01.03.26.ipynb
Run from anywhere:  python statistical_analysis.py
Outputs -> ./outputs
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "TPMS_data_27.12.25.xlsx"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

FAMILIES = ["Diamond", "Gyroid", "Lidinoid", "Schwarz", "SplitP"]
COLORS = dict(zip(FAMILIES, ["#4C84C0", "#55AD6B", "#DC4B52", "#8E5AA6", "#E8893A"]))
XL_E = "E (MPa)"
XL_S = r"E$_{\mathrm{specific}}$ (MPa/(kg/m$^3$))"
BINS = 18

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 13,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True})

df = pd.read_excel(DATA).rename(columns={
    "Lattice Type": "family", "Thickness (t) (mm)": "t", "UCx (mm)": "UCx",
    "UCy (mm)": "UCy", "UCz (mm)": "UCz", "Young's Modulus (MPa)": "E",
    "Specific Stiffness (MPa/(kg/m3))": "Es"})
assert len(df) == 270 and (df.family.value_counts() == 54).all()

# ---- tables -------------------------------------------------------------
overall = df[["E", "Es"]].agg(["count", "mean", "std", "min", "median", "max"]).T
by_fam = df.groupby("family")[["E", "Es"]].agg(["mean", "std", "min", "max"])
overall.to_csv(OUT / "table_overall.csv")
by_fam.to_csv(OUT / "table_family.csv")
print(overall.round(4), "\n", by_fam.round(3))

# ---- figures --------------------------------------------------------------
def hist(ax, v, color, xlabel, bins=BINS):
    ax.hist(v, bins=bins, color=color, alpha=0.6, edgecolor=color, hatch="xx", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")

# S3: all 270 designs
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
hist(ax[0], df.E, "#3CA55C", XL_E); hist(ax[1], df.Es, "#8E44AD", XL_S)
for a, l in zip(ax, "ab"): a.text(0.0, 1.03, f"{l})", transform=a.transAxes, fontweight="bold")
fig.savefig(OUT / "FigS3_hist_all.png", dpi=300); plt.close(fig)

# S1 / S2: per-family histograms
for key, xl, name in (("E", XL_E, "FigS1_hist_E_by_family"), ("Es", XL_S, "FigS2_hist_Es_by_family")):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for a, fam, l in zip(axes.flat, FAMILIES, "abcde"):
        hist(a, df.loc[df.family == fam, key], COLORS[fam], xl)
        a.text(0.0, 1.03, f"{l}) {fam}", transform=a.transAxes, fontweight="bold")
    axes.flat[-1].axis("off")
    fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)

# S4: box plots (full axis range, nothing clipped)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
rng = np.random.default_rng(0)
for a, key, yl, l in zip(ax, ("E", "Es"), (XL_E, XL_S), "ab"):
    data = [df.loc[df.family == f, key].values for f in FAMILIES]
    bp = a.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                   medianprops=dict(color="white", lw=2))
    for p, f in zip(bp["boxes"], FAMILIES): p.set_facecolor(COLORS[f]); p.set_alpha(0.85)
    for i, d in enumerate(data, 1): a.scatter(rng.normal(i, 0.08, len(d)), d, s=14, c="k", alpha=0.3, lw=0)
    a.set_xticks(range(1, 6)); a.set_xticklabels(FAMILIES, rotation=20, ha="right")
    a.set_ylabel(yl); a.text(0.0, 1.03, f"{l})", transform=a.transAxes, fontweight="bold")
fig.savefig(OUT / "FigS4_boxplots.png", dpi=300); plt.close(fig)
print("saved to", OUT)
