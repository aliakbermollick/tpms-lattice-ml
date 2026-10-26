"""Family-wise SHAP (KernelSHAP) of the Stacked Ensemble models for E and E_specific (manuscript Sec. 2.8/3.6, Fig. 4, Suppl. Figs. S16).
Extracted from SHAP SE 06.03.26.ipynb (cell 8); only paths changed.
KernelSHAP, 100-point random background, nsamples='auto', all 270 designs explained, aggregated per family (n=54).
Needs scikit-learn==1.5.2 to load the saved models. Set TPMS_MODEL_RUN to the folder
TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46 (default: ../03_ml_models/outputs/<that name>).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_RUN = os.environ.get("TPMS_MODEL_RUN", os.path.join(HERE, "..", "03_ml_models", "outputs", "TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46"))
# ================================================================
# Family-wise SHAP analysis (SE models)
# - KernelSHAP; bar plots (mean |SHAP|) and beeswarm plots for each family
# - Arial font, subscripted UC labels, kg/m^3 superscript
# - Figures saved as PNG + PDF + SVG (editable text)
# - Saves the family-wise mean |SHAP| values (Excel) and a runtime summary
# ================================================================

import os, ast, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import shap
os.environ["TQDM_NOTEBOOK"] = "0"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm

from joblib import load

# =========================
# 1) FONT (Arial, Windows)
# =========================
ARIAL_PATH = r"C:\Windows\Fonts\arial.ttf"
if not os.path.exists(ARIAL_PATH):
    raise FileNotFoundError(f"Arial font not found at: {ARIAL_PATH}")

fm.fontManager.addfont(ARIAL_PATH)
arial_name = fm.FontProperties(fname=ARIAL_PATH).get_name()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [arial_name],
    "pdf.fonttype": 42,     # editable text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none", # editable text in SVG
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.default": "regular",
})

print("Forced font:", arial_name)

# =========================
# 2) PATHS (SE MODELS + DATA)
# =========================
DATA_XLSX = os.path.join(HERE, "..", "data", "TPMS_data_27.12.25.xlsx")

SE_YM = os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_StackedEnsemble_Youngs_Modulus_MPa.pkl")

SE_SS = os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_StackedEnsemble_Specific_Stiffness_MPa-kg-m3.pkl")

OUT_ROOT = os.path.join(HERE, "outputs")

# =========================
# 3) CONFIG
# =========================
ENC_COL  = "Structure Encoding"
NUM_COLS = ["Thickness (t) (mm)", "UCx (mm)", "UCy (mm)", "UCz (mm)"]

RANDOM_STATE = 42
KERNEL_BACKGROUND_N = 100
NSAMPLES = "auto"

RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join(OUT_ROOT, f"shap_se_ARIAL_{RUN_STAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

# =========================
# 4) FAMILY DISPLAY (SplitP)
# =========================
def norm_key(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())

PANEL_ORDER_DISPLAY = ["Schwarz", "Lidinoid", "Diamond", "Gyroid", "SplitP"]

DISPLAY_MAP = {
    "schwarz": "Schwarz",
    "lidinoid": "Lidinoid",
    "diamond": "Diamond",
    "gyroid": "Gyroid",
    "splitp": "SplitP",
}

# =========================
# 5) LABELS
# =========================
XTICKS = [r"$t$", r"UC$_x$", r"UC$_y$", r"UC$_z$"]
YTICKS = [r"UC$_z$ (mm)", r"UC$_y$ (mm)", r"UC$_x$ (mm)", r"$t$ (mm)"]

SS_UNIT = r"MPa/(kg/m$^{3}$)"
YM_TITLE = "Young's Modulus (MPa)"
SS_TITLE = f"Specific Stiffness ({SS_UNIT})"

# =========================
# 6) HELPERS
# =========================
def ensure_arraylike(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, str):
        try:
            return list(ast.literal_eval(v))
        except Exception:
            return None
    return None

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_png_pdf_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".svg", bbox_inches="tight", facecolor="white")

def _get_feature_names(bundle):
    if isinstance(bundle, dict):
        if "metadata" in bundle and isinstance(bundle["metadata"], dict):
            meta = bundle["metadata"]
            if "feature_names" in meta:
                return list(meta["feature_names"])
        if "feature_names" in bundle:
            return list(bundle["feature_names"])
    raise KeyError("Could not find feature_names in model bundle.")

def _get_family_order(bundle):
    if isinstance(bundle, dict):
        if "metadata" in bundle and isinstance(bundle["metadata"], dict):
            meta = bundle["metadata"]
            if "family_order" in meta:
                return list(meta["family_order"])
    raise KeyError("Could not find family_order in model bundle metadata.")

def _get_predictor_from_bundle(bundle):
    if not isinstance(bundle, dict):
        if hasattr(bundle, "predict"):
            return bundle
        raise TypeError("Loaded model is neither a dict bundle nor a predictor object with .predict().")

    candidate_keys = [
        "pipeline",
        "model",
        "estimator",
        "regressor",
        "final_model",
        "stacked_model",
        "meta_model",
    ]

    for key in candidate_keys:
        if key in bundle and hasattr(bundle[key], "predict"):
            return bundle[key]

    for _, val in bundle.items():
        if hasattr(val, "predict"):
            return val

    raise KeyError("Could not find predictor with .predict() inside the bundle.")

def load_model_bundle_general(path):
    b = load(path)
    predictor = _get_predictor_from_bundle(b)
    feature_names = _get_feature_names(b)
    family_order = _get_family_order(b)

    preproc = None
    final_est = predictor

    if hasattr(predictor, "steps") and len(predictor.steps) >= 2:
        try:
            preproc = predictor[:-1]
            final_est = predictor.steps[-1][1]
        except Exception:
            preproc = None
            final_est = predictor

    meta = {
        "feature_names": feature_names,
        "family_order": family_order,
    }
    return predictor, preproc, final_est, meta

def decode_family_from_onehot(enc, order):
    enc = np.asarray(enc, dtype=float)
    if enc.size == 0:
        return None
    fam = order[int(np.argmax(enc))]
    return DISPLAY_MAP.get(norm_key(fam), str(fam))

def format_elapsed_time(seconds):
    seconds = float(seconds)
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    total_seconds_int = int(round(seconds))
    h = total_seconds_int // 3600
    m = (total_seconds_int % 3600) // 60
    s = total_seconds_int % 60
    return {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "hms": f"{h:02d}:{m:02d}:{s:02d}"
    }

def compute_familywise_bar_values(shap_vals, fam_eval, feature_names):
    num_idx = [feature_names.index(c) for c in NUM_COLS]

    rows = []
    for fam_disp in PANEL_ORDER_DISPLAY:
        idx = np.where(np.asarray(fam_eval) == fam_disp)[0]

        if len(idx) > 0:
            vals = np.abs(shap_vals[idx][:, num_idx]).mean(axis=0)
        else:
            vals = np.zeros(len(num_idx))

        rows.append({
            "Family": fam_disp,
            "t_mean_abs_SHAP": vals[0],
            "UCx_mean_abs_SHAP": vals[1],
            "UCy_mean_abs_SHAP": vals[2],
            "UCz_mean_abs_SHAP": vals[3],
        })

    return pd.DataFrame(rows)

# =========================
# 7) PLOTS
# =========================
def familywise_bar_panels(shap_vals, fam_eval, feature_names, out_dir):
    num_idx = [feature_names.index(c) for c in NUM_COLS]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)

    for ax, fam_disp in zip(axes, PANEL_ORDER_DISPLAY):
        idx = np.where(np.asarray(fam_eval) == fam_disp)[0]
        vals = np.abs(shap_vals[idx][:, num_idx]).mean(axis=0) if len(idx) > 0 else np.zeros(len(num_idx))

        ax.bar(range(4), vals)
        ax.set_title(fam_disp)
        ax.set_xticks(range(4))
        ax.set_xticklabels(XTICKS)
        ax.grid(False)

    fig.supylabel(r"mean $|$SHAP$|$")
    fig.tight_layout()

    save_png_pdf_svg(fig, os.path.join(out_dir, "familywise_bars"))
    plt.close(fig)

def familywise_beeswarm_panels(shap_vals, X_eval_trans_df, fam_eval, feature_names, out_dir):
    desired = ["UCz (mm)", "UCy (mm)", "UCx (mm)", "Thickness (t) (mm)"]
    num_idx = [feature_names.index(c) for c in desired]

    Xn = X_eval_trans_df.iloc[:, num_idx].values
    vmin, vmax = np.percentile(Xn, 2), np.percentile(Xn, 98)

    cmap = plt.cm.coolwarm
    rng_local = np.random.RandomState(RANDOM_STATE)

    fig, axes = plt.subplots(len(PANEL_ORDER_DISPLAY), 1, figsize=(7, 10.5), sharex=True)

    for ax, fam_disp in zip(axes, PANEL_ORDER_DISPLAY):
        idx = np.where(np.asarray(fam_eval) == fam_disp)[0]
        ax.axvline(0, color="0.5", lw=1)

        if len(idx) > 0:
            S = shap_vals[idx][:, num_idx]
            F = X_eval_trans_df.iloc[idx, num_idx].values
            norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2.0, vmax=vmax)

            for j in range(4):
                x = S[:, j]
                c = F[:, j]
                y = j + (rng_local.rand(len(x)) - 0.5) * 0.25
                ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=22, alpha=0.9, edgecolor="none")

        ax.set_yticks(range(4))
        ax.set_yticklabels(YTICKS)
        ax.set_title(fam_disp, loc="left", fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("SHAP Value")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Feature value (scaled)")

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    save_png_pdf_svg(fig, os.path.join(out_dir, "familywise_beeswarm"))
    plt.close(fig)

# =========================
# 8) RUN
# =========================
overall_t0 = time.time()

for p in [DATA_XLSX, SE_YM, SE_SS, OUT_ROOT]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"NOT FOUND: {p}")

df = pd.read_excel(DATA_XLSX)

targets = [
    (YM_TITLE, SE_YM),
    (SS_TITLE, SE_SS),
]

combined_excel_path = os.path.join(RUN_DIR, "Familywise_Mean_Abs_SHAP_Values.xlsx")

with pd.ExcelWriter(combined_excel_path, engine="openpyxl") as writer:
    for title, model_path in targets:
        target_t0 = time.time()

        print("\n" + "="*70)
        print("TARGET:", title)
        print("MODEL :", model_path)

        predictor, preproc, final_est, meta = load_model_bundle_general(model_path)
        feature_names = list(meta["feature_names"])
        family_order  = list(meta["family_order"])

        work = df.copy()
        work[ENC_COL] = work[ENC_COL].apply(ensure_arraylike)
        work = work.dropna(subset=[ENC_COL] + NUM_COLS)

        for c in NUM_COLS:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.dropna(subset=NUM_COLS)

        enc_len = len(family_order)

        def pad_or_trim(e, want):
            e = list(e)
            if len(e) < want:
                e += [0] * (want - len(e))
            if len(e) > want:
                e = e[:want]
            return e

        enc_mat = np.vstack(work[ENC_COL].apply(lambda e: pad_or_trim(e, enc_len)).values).astype(float)
        num_mat = work[NUM_COLS].values.astype(float)

        X_full = np.hstack([enc_mat, num_mat])

        if X_full.shape[1] == len(feature_names):
            X_full_df = pd.DataFrame(X_full, columns=feature_names)
        else:
            enc_names_guess = feature_names[:enc_len]
            X_full_df = pd.DataFrame(X_full, columns=enc_names_guess + NUM_COLS)
            X_full_df = X_full_df.reindex(columns=feature_names)

        fam_labels = np.array([decode_family_from_onehot(e, family_order) for e in enc_mat])

        if preproc is not None:
            X_trans = preproc.transform(X_full_df.values)
            X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
            pred_fn = final_est.predict
        else:
            X_trans_df = X_full_df.copy()
            pred_fn = predictor.predict

        rng = np.random.RandomState(RANDOM_STATE)
        bg_n = min(KERNEL_BACKGROUND_N, len(X_trans_df))
        bg_idx = rng.choice(len(X_trans_df), size=bg_n, replace=False)
        background = X_trans_df.iloc[bg_idx]

        explainer = shap.KernelExplainer(pred_fn, background, link="identity")
        shap_vals = explainer.shap_values(X_trans_df, nsamples=NSAMPLES, silent=True)

        if isinstance(shap_vals, list):
            shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0]

        tgt_dir = safe_mkdir(os.path.join(RUN_DIR, title.replace(" ", "_").replace("/", "_")))
        print("Saving to:", tgt_dir)

        # Bar and beeswarm plots (no titles)
        familywise_bar_panels(
            shap_vals, fam_labels, feature_names, tgt_dir
        )

        familywise_beeswarm_panels(
            shap_vals, X_trans_df, fam_labels, feature_names, tgt_dir
        )

        # Exact values behind the bar plot
        barvals_df = compute_familywise_bar_values(
            shap_vals, fam_labels, feature_names
        )

        # Save per-target Excel inside target folder
        excel_name = f"{title.replace(' ', '_').replace('/', '_')}_familywise_mean_abs_SHAP.xlsx"
        barvals_path = os.path.join(tgt_dir, excel_name)
        barvals_df.to_excel(barvals_path, index=False)

        # Save to combined Excel workbook in RUN_DIR
        sheet_name = "YM_familywise_mean_abs_SHAP" if title == YM_TITLE else "SS_familywise_mean_abs_SHAP"
        barvals_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print("\nFamily-wise mean |SHAP| values:")
        print(barvals_df.to_string(index=False))
        print("\nExcel saved:")
        print(barvals_path)

        target_elapsed = format_elapsed_time(time.time() - target_t0)
        print("\nSaved: familywise_bars(.png/.pdf/.svg), familywise_beeswarm(.png/.pdf/.svg)")
        print(f"Target runtime: {target_elapsed['hms']}  |  {target_elapsed['minutes']:.2f} min")

overall_elapsed = format_elapsed_time(time.time() - overall_t0)

runtime_txt = os.path.join(RUN_DIR, "RUN_TIME_SUMMARY.txt")
with open(runtime_txt, "w", encoding="utf-8") as f:
    f.write("SHAP SE Runtime Summary\n")
    f.write("=======================\n")
    f.write(f"Run folder: {RUN_DIR}\n")
    f.write(f"Dataset: {DATA_XLSX}\n")
    f.write(f"YM model: {SE_YM}\n")
    f.write(f"SS model: {SE_SS}\n")
    f.write(f"Combined SHAP excel: {combined_excel_path}\n")
    f.write(f"Elapsed time (HH:MM:SS): {overall_elapsed['hms']}\n")
    f.write(f"Elapsed time (seconds): {overall_elapsed['seconds']:.2f}\n")
    f.write(f"Elapsed time (minutes): {overall_elapsed['minutes']:.2f}\n")
    f.write(f"Elapsed time (hours): {overall_elapsed['hours']:.2f}\n")

print("\n=== DONE ===")
print("RUN_DIR:", RUN_DIR)
print("\nCombined Excel saved:")
print(combined_excel_path)
print("\n=== TOTAL RUN TIME ===")
print(f"HH:MM:SS  : {overall_elapsed['hms']}")
print(f"Seconds   : {overall_elapsed['seconds']:.2f}")
print(f"Minutes   : {overall_elapsed['minutes']:.2f}")
print(f"Hours     : {overall_elapsed['hours']:.2f}")
print("\nSaved runtime summary:")
print(runtime_txt)