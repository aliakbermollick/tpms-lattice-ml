"""Family-wise Spearman rank correlation (manuscript Section 2.4 / 3.2, Supplementary Fig. S5).

Original notebook: Correlation Matrix 01.03.26.ipynb
Run:  python spearman_correlation.py     Outputs -> ./outputs
Optional --verify <dir>: compare against previously saved CorrMatrix_*_Spearman.csv files.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "TPMS_data_27.12.25.xlsx"
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

sns.set_style("white")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 13})

COLS = ["Thickness (t) (mm)", "UCx (mm)", "UCy (mm)", "UCz (mm)",
        "Young's Modulus (MPa)", "Specific Stiffness (MPa/(kg/m3))"]
LABELS = {COLS[0]: r"Thickness $t$ (mm)", COLS[1]: r"$UC_{x}$ (mm)", COLS[2]: r"$UC_{y}$ (mm)",
          COLS[3]: r"$UC_{z}$ (mm)", COLS[4]: r"Young's modulus $E$ (MPa)",
          COLS[5]: r"Specific stiffness $E_{\mathrm{specific}}$" + "\n" + r"(MPa/(kg/m$^{3}$))"}
FAMILIES = ["Diamond", "Gyroid", "Lidinoid", "Schwarz", "SplitP"]

df = pd.read_excel(DATA)
df["Lattice Type"] = df["Lattice Type"].astype(str).str.strip()
mats = {}
for fam in FAMILIES:
    m = df.loc[df["Lattice Type"] == fam, COLS].corr(method="spearman").round(2).replace(-0.00, 0.00)
    mats[fam] = m
    m.to_csv(OUT / f"CorrMatrix_{fam}_Spearman.csv")
    fig, ax = plt.subplots(figsize=(9.3, 7.8))
    sns.heatmap(m.rename(index=LABELS, columns=LABELS), ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.4, vmin=-1, vmax=1, cbar_kws={"shrink": 0.82},
                annot_kws={"fontsize": 13, "color": "black"})
    ax.set_title(f"Spearman Correlation Matrix: {fam}", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / f"Heatmap_{fam}_Spearman.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if "--verify" in sys.argv:
    ref = Path(sys.argv[sys.argv.index("--verify") + 1])
    for fam in FAMILIES:
        old = pd.read_csv(ref / f"CorrMatrix_{fam}_Spearman.csv", index_col=0)
        print(fam, "max |diff| vs saved:", float((old.values - mats[fam].values).__abs__().max()))
