"""Inverse design for specific stiffness (Sec. 2.7/3.5, Table 7, Suppl. S5).
Extracted from prediction 01.03.26.ipynb (cell 2); only paths changed.
Needs scikit-learn==1.5.2 to load the saved models. Set TPMS_MODEL_RUN to the folder
TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46 (default: ../03_ml_models/outputs/<that name>).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_RUN = os.environ.get("TPMS_MODEL_RUN", os.path.join(HERE, "..", "03_ml_models", "outputs", "TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46"))
# ============================================================
# Inverse design for a target specific stiffness (GPR and Stacked Ensemble)
#
# - Continuous search: Latin Hypercube sampling followed by local refinement
# - Candidates are ranked by the specific-stiffness target error
# - The geometry is rounded to the manufacturing grid and re-predicted
# - The output folder is created with retries (OneDrive-safe) and named
#   TPMS_InverseDesign_TargetSS_YYYYMMDD_HHMMSS
# ============================================================

import os, re, shutil, tempfile, time
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime

# SciPy LHS if available
try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except Exception:
    _HAS_SCIPY_QMC = False


# -----------------------------
# 0) MODEL PATHS (GPR and Stacked Ensemble)
# -----------------------------
YM_GPR_PATH = os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_GPR_Youngs_Modulus_MPa.pkl")
YM_STACK_PATH = os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_StackedEnsemble_Youngs_Modulus_MPa.pkl")

SS_GPR_PATH = os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_GPR_Specific_Stiffness_MPa-kg-m3.pkl")
SS_STACK_PATH = os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_StackedEnsemble_Specific_Stiffness_MPa-kg-m3.pkl")


# -----------------------------
# 1) INVERSE DESIGN TARGETS (Specific Stiffness targets)
# -----------------------------
TARGETS_SS = [
    ("Diamond", 0.68),
    ("Gyroid", 0.45),
    ("Lidinoid", 0.70),
    ("Schwarz", 0.62),
    ("Split P", 0.75),
]

# Training configuration (must match the training script)
FAMILY_ORDER = ["Diamond", "Gyroid", "Split P", "Schwarz", "Lidinoid"]
FAMILY_NAME_COL = "Lattice Type"
NUM_COLS = ["Thickness (t) (mm)", "UCx (mm)", "UCy (mm)", "UCz (mm)"]

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------
# 2) HARD LIMITS
# -----------------------------
UC_HARD_LO, UC_HARD_HI = 2.00, 4.00
T_HARD_LO,  T_HARD_HI  = 0.30, 0.40

# -----------------------------
# 3) ROUNDING (manufacturable suggestion ONLY)
# -----------------------------
ROUND_UC_DECIMALS = 2
ROUND_T_STEP      = 0.01

# -----------------------------
# 4) SEARCH SETTINGS
# -----------------------------
N_GLOBAL_LHS       = 60000
REFINE_ROUNDS      = 5
SAMPLES_PER_ROUND  = 48000
KEEP_TOP           = 400
FINAL_TOP_K        = 30

SIGMA0_FRAC = dict(UCx=0.10, UCy=0.10, UCz=0.10, t=0.10)
SIGMA_SHRINK = 0.55


# ============================================================
# 5) IO HELPERS (ROBUST FOLDER CREATION)
# ============================================================

def ensure_dir(path: str, retries: int = 5, wait_s: float = 0.25):
    """
    Robust mkdir for OneDrive paths. Retries + verifies existence.
    Falls back to a local Documents temp folder if needed.
    """
    path = os.path.abspath(path)
    last_err = None
    for _ in range(retries):
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.isdir(path):
                return path
        except Exception as e:
            last_err = e
        time.sleep(wait_s)

    # fallback
    fallback_root = os.path.join(os.path.expanduser("~"), "Documents")
    fallback = os.path.join(fallback_root, os.path.basename(path))
    for _ in range(retries):
        try:
            os.makedirs(fallback, exist_ok=True)
            if os.path.isdir(fallback):
                print(f"[WARN] Could not create OneDrive folder. Using fallback:\n{fallback}")
                return fallback
        except Exception as e:
            last_err = e
        time.sleep(wait_s)

    raise RuntimeError(f"Could not create output directory.\nTried:\n{path}\nFallback:\n{fallback}\nLast error: {last_err}")


# ============================================================
# 6) ML + SEARCH HELPERS
# ============================================================

def load_bundle(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model not found:\n{pkl_path}")
    b = load(pkl_path)
    if not isinstance(b, dict) or "pipeline" not in b:
        raise ValueError(f"Unexpected bundle format in:\n{pkl_path}\nExpected dict with key 'pipeline'.")
    return b

def safe_read_excel_via_temp_copy(xlsx_path, sheet_name=0):
    """Avoid OneDrive/Excel locks by copying to a temp location first."""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")
    tmp_dir = tempfile.mkdtemp(prefix="tpms_excel_")
    tmp_xlsx = os.path.join(tmp_dir, os.path.basename(xlsx_path))
    try:
        shutil.copy2(xlsx_path, tmp_xlsx)
        df_local = pd.read_excel(tmp_xlsx, sheet_name=sheet_name, engine="openpyxl")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return df_local

def canon_family(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).strip().lower()
    s0 = s0.replace("_", " ").replace("-", " ")
    s0 = re.sub(r"\s+", " ", s0).strip()
    if "diamond" in s0: return "Diamond"
    if "gyroid" in s0: return "Gyroid"
    if "lidinoid" in s0: return "Lidinoid"
    if "schwarz" in s0: return "Schwarz"
    if ("split" in s0 and "p" in s0) or s0 in ["splitp","split p"]:
        return "Split P"
    return str(s).strip()

def one_hot_family(family: str, order=FAMILY_ORDER):
    fam = canon_family(family)
    if fam not in order:
        raise ValueError(f"Family '{family}' canonicalized to '{fam}', not in FAMILY_ORDER = {order}")
    v = np.zeros(len(order), dtype=float)
    v[order.index(fam)] = 1.0
    return v

def make_feature_matrix(family: str, UCx, UCy, UCz, t):
    """X = one-hot + [t, UCx, UCy, UCz]"""
    UCx = np.asarray(UCx, dtype=float).ravel()
    UCy = np.asarray(UCy, dtype=float).ravel()
    UCz = np.asarray(UCz, dtype=float).ravel()
    t   = np.asarray(t, dtype=float).ravel()
    n = len(UCx)
    enc = np.tile(one_hot_family(family), (n, 1))
    num = np.column_stack([t, UCx, UCy, UCz])
    return np.hstack([enc, num])

def clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

def round_geometry(UCx, UCy, UCz, t):
    UCx_r = np.round(np.asarray(UCx, float), ROUND_UC_DECIMALS)
    UCy_r = np.round(np.asarray(UCy, float), ROUND_UC_DECIMALS)
    UCz_r = np.round(np.asarray(UCz, float), ROUND_UC_DECIMALS)
    t_r   = np.round(np.asarray(t, float) / ROUND_T_STEP) * ROUND_T_STEP

    UCx_r = clip(UCx_r, UC_HARD_LO, UC_HARD_HI)
    UCy_r = clip(UCy_r, UC_HARD_LO, UC_HARD_HI)
    UCz_r = clip(UCz_r, UC_HARD_LO, UC_HARD_HI)
    t_r   = clip(t_r,   T_HARD_LO,  T_HARD_HI)
    return UCx_r, UCy_r, UCz_r, t_r

def lhs_sample(n, bounds, rng):
    lo = np.array([bounds["UCx"][0], bounds["UCy"][0], bounds["UCz"][0], bounds["t"][0]], float)
    hi = np.array([bounds["UCx"][1], bounds["UCy"][1], bounds["UCz"][1], bounds["t"][1]], float)

    if _HAS_SCIPY_QMC:
        sampler = qmc.LatinHypercube(d=4, seed=int(rng.integers(1, 10**9)))
        u = sampler.random(n=n)
        x = qmc.scale(u, lo, hi)
    else:
        u = np.empty((n, 4), float)
        for j in range(4):
            cut = (np.arange(n) + rng.random(n)) / n
            rng.shuffle(cut)
            u[:, j] = cut
        x = lo + u * (hi - lo)

    UCx, UCy, UCz, t = x[:,0], x[:,1], x[:,2], x[:,3]
    UCx = clip(UCx, UC_HARD_LO, UC_HARD_HI)
    UCy = clip(UCy, UC_HARD_LO, UC_HARD_HI)
    UCz = clip(UCz, UC_HARD_LO, UC_HARD_HI)
    t   = clip(t,   T_HARD_LO,  T_HARD_HI)
    return UCx, UCy, UCz, t

def evaluate_candidates_ss(family, target_ss, ym_model, ss_model, UCx, UCy, UCz, t):
    X = make_feature_matrix(family, UCx, UCy, UCz, t)
    ss_pred = ss_model.predict(X).astype(float).ravel()
    ym_pred = ym_model.predict(X).astype(float).ravel()

    df = pd.DataFrame({
        "UCx": UCx, "UCy": UCy, "UCz": UCz, "t": t,
        "SS_pred": ss_pred, "YM_pred": ym_pred
    })
    df["abs_err_ss"] = np.abs(df["SS_pred"] - target_ss)
    df["SS_error_%"] = (df["abs_err_ss"] / (target_ss if target_ss != 0 else np.nan)) * 100.0
    return df

def get_family_bounds(df_train, family, qlo=0.01, qhi=0.99, min_rows=10):
    d = df_train.copy()
    d["_family_can_"] = d[FAMILY_NAME_COL].apply(canon_family)
    fam_can = canon_family(family)
    dff = d[d["_family_can_"] == fam_can].copy()

    for c in NUM_COLS:
        dff[c] = pd.to_numeric(dff[c], errors="coerce")
    dff = dff.dropna(subset=NUM_COLS)

    if len(dff) >= min_rows:
        t_lo, t_hi = dff["Thickness (t) (mm)"].quantile([qlo, qhi]).values
        x_lo, x_hi = dff["UCx (mm)"].quantile([qlo, qhi]).values
        y_lo, y_hi = dff["UCy (mm)"].quantile([qlo, qhi]).values
        z_lo, z_hi = dff["UCz (mm)"].quantile([qlo, qhi]).values
        b = {"t":(float(t_lo),float(t_hi)), "UCx":(float(x_lo),float(x_hi)),
             "UCy":(float(y_lo),float(y_hi)), "UCz":(float(z_lo),float(z_hi))}
        return b, f"per-family quantiles (n={len(dff)})"

    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=NUM_COLS)
    t_lo, t_hi = d["Thickness (t) (mm)"].quantile([qlo, qhi]).values
    x_lo, x_hi = d["UCx (mm)"].quantile([qlo, qhi]).values
    y_lo, y_hi = d["UCy (mm)"].quantile([qlo, qhi]).values
    z_lo, z_hi = d["UCz (mm)"].quantile([qlo, qhi]).values
    b = {"t":(float(t_lo),float(t_hi)), "UCx":(float(x_lo),float(x_hi)),
         "UCy":(float(y_lo),float(y_hi)), "UCz":(float(z_lo),float(z_hi))}
    return b, "fallback: global bounds"

def apply_hard_limits(bounds: dict):
    b = dict(bounds)
    b["UCx"] = (max(b["UCx"][0], UC_HARD_LO), min(b["UCx"][1], UC_HARD_HI))
    b["UCy"] = (max(b["UCy"][0], UC_HARD_LO), min(b["UCy"][1], UC_HARD_HI))
    b["UCz"] = (max(b["UCz"][0], UC_HARD_LO), min(b["UCz"][1], UC_HARD_HI))
    b["t"]   = (max(b["t"][0],   T_HARD_LO),  min(b["t"][1],   T_HARD_HI))
    for k,(lo,hi) in b.items():
        if lo >= hi:
            b[k] = (UC_HARD_LO, UC_HARD_HI) if k in ["UCx","UCy","UCz"] else (T_HARD_LO, T_HARD_HI)
    return b

def add_rounded_predictions_ss(df, family, target_ss, ym_model, ss_model):
    Xr = make_feature_matrix(
        family,
        df["UCx_rounded"].values,
        df["UCy_rounded"].values,
        df["UCz_rounded"].values,
        df["t_rounded"].values
    )
    ss_r = ss_model.predict(Xr).astype(float).ravel()
    ym_r = ym_model.predict(Xr).astype(float).ravel()

    out = df.copy()
    out["SS_pred_rounded"] = ss_r
    out["YM_pred_rounded"] = ym_r
    out["abs_err_ss_rounded"] = np.abs(out["SS_pred_rounded"] - target_ss)
    out["SS_error_%_rounded"] = (out["abs_err_ss_rounded"] / (target_ss if target_ss != 0 else np.nan)) * 100.0
    return out

def inverse_design_lhs_continuous_ss(
    family: str,
    target_ss: float,
    ym_model,
    ss_model,
    bounds: dict,
    model_name: str,
    bounds_source: str
):
    stage_rows = []
    evaluated_set = set()

    def _key(row4):
        return (round(float(row4[0]), 6), round(float(row4[1]), 6),
                round(float(row4[2]), 6), round(float(row4[3]), 6))

    # Stage 0: Global LHS
    UCx, UCy, UCz, t = lhs_sample(N_GLOBAL_LHS, bounds, rng)
    df0 = evaluate_candidates_ss(family, target_ss, ym_model, ss_model, UCx, UCy, UCz, t)

    for r4 in df0[["UCx","UCy","UCz","t"]].to_numpy():
        evaluated_set.add(_key(r4))

    best0 = df0.sort_values(["SS_error_%","abs_err_ss"]).iloc[0]

    stage_rows.append({
        "Model": model_name,
        "Bounds_Source": bounds_source,
        "Family": canon_family(family),
        "Stage": "Global (LHS)",
        "Round": 0,
        "Requested_Samples": int(N_GLOBAL_LHS),
        "Unique_Geometry_Evaluated": int(len(df0)),
        "Cumulative_Unique_Evaluated": int(len(evaluated_set)),
        "Best_SS_error_%": float(best0["SS_error_%"]),
        "Best_abs_err_ss": float(best0["abs_err_ss"]),
        "Best_SS_pred": float(best0["SS_pred"]),
        "Best_YM_pred": float(best0["YM_pred"]),
    })

    pool = df0.sort_values(["SS_error_%","abs_err_ss"]).head(KEEP_TOP).copy().reset_index(drop=True)

    sig = {
        "UCx": SIGMA0_FRAC["UCx"]*(bounds["UCx"][1]-bounds["UCx"][0]),
        "UCy": SIGMA0_FRAC["UCy"]*(bounds["UCy"][1]-bounds["UCy"][0]),
        "UCz": SIGMA0_FRAC["UCz"]*(bounds["UCz"][1]-bounds["UCz"][0]),
        "t":   SIGMA0_FRAC["t"]  *(bounds["t"][1]-bounds["t"][0]),
    }

    for rr in range(1, REFINE_ROUNDS+1):
        for k in sig:
            sig[k] *= SIGMA_SHRINK

        base = pool.sample(n=min(len(pool), KEEP_TOP), replace=False, random_state=RANDOM_SEED + rr)
        n_per = int(np.ceil(SAMPLES_PER_ROUND / len(base)))

        UCx_new = np.repeat(base["UCx"].values, n_per) + rng.normal(0, sig["UCx"], size=len(base)*n_per)
        UCy_new = np.repeat(base["UCy"].values, n_per) + rng.normal(0, sig["UCy"], size=len(base)*n_per)
        UCz_new = np.repeat(base["UCz"].values, n_per) + rng.normal(0, sig["UCz"], size=len(base)*n_per)
        t_new   = np.repeat(base["t"].values,   n_per) + rng.normal(0, sig["t"],   size=len(base)*n_per)

        UCx_new = UCx_new[:SAMPLES_PER_ROUND]
        UCy_new = UCy_new[:SAMPLES_PER_ROUND]
        UCz_new = UCz_new[:SAMPLES_PER_ROUND]
        t_new   = t_new[:SAMPLES_PER_ROUND]

        UCx_new = clip(UCx_new, UC_HARD_LO, UC_HARD_HI)
        UCy_new = clip(UCy_new, UC_HARD_LO, UC_HARD_HI)
        UCz_new = clip(UCz_new, UC_HARD_LO, UC_HARD_HI)
        t_new   = clip(t_new,   T_HARD_LO,  T_HARD_HI)

        dfR = evaluate_candidates_ss(family, target_ss, ym_model, ss_model, UCx_new, UCy_new, UCz_new, t_new)

        for r4 in dfR[["UCx","UCy","UCz","t"]].to_numpy():
            evaluated_set.add(_key(r4))

        pool = pd.concat([pool, dfR], ignore_index=True)
        pool = pool.sort_values(["SS_error_%","abs_err_ss"]).head(KEEP_TOP).reset_index(drop=True)

        bestR = pool.iloc[0]

        stage_rows.append({
            "Model": model_name,
            "Bounds_Source": bounds_source,
            "Family": canon_family(family),
            "Stage": "Local refinement",
            "Round": int(rr),
            "Requested_Samples": int(SAMPLES_PER_ROUND),
            "Unique_Geometry_Evaluated": int(len(dfR)),
            "Cumulative_Unique_Evaluated": int(len(evaluated_set)),
            "Best_SS_error_%": float(bestR["SS_error_%"]),
            "Best_abs_err_ss": float(bestR["abs_err_ss"]),
            "Best_SS_pred": float(bestR["SS_pred"]),
            "Best_YM_pred": float(bestR["YM_pred"]),
        })

    top = pool.sort_values(["SS_error_%","abs_err_ss"]).head(FINAL_TOP_K).copy().reset_index(drop=True)
    top.insert(0, "Family", canon_family(family))
    top.insert(1, "Target_SS", float(target_ss))
    top.insert(0, "Model", model_name)
    top.insert(2, "Bounds_Source", bounds_source)

    UCx_r, UCy_r, UCz_r, t_r = round_geometry(top["UCx"].values, top["UCy"].values, top["UCz"].values, top["t"].values)
    top["UCx_rounded"] = UCx_r
    top["UCy_rounded"] = UCy_r
    top["UCz_rounded"] = UCz_r
    top["t_rounded"]   = t_r

    return top, pd.DataFrame(stage_rows), len(evaluated_set)


# ============================================================
# 7) LOAD MODELS + TRAINING DATA
# ============================================================

ym_gpr_bundle   = load_bundle(YM_GPR_PATH)
ym_stack_bundle = load_bundle(YM_STACK_PATH)
ss_gpr_bundle   = load_bundle(SS_GPR_PATH)
ss_stack_bundle = load_bundle(SS_STACK_PATH)

YM_GPR   = ym_gpr_bundle["pipeline"]
YM_STACK = ym_stack_bundle["pipeline"]
SS_GPR   = ss_gpr_bundle["pipeline"]
SS_STACK = ss_stack_bundle["pipeline"]

DATA_XLSX = ss_gpr_bundle.get("metadata", {}).get("data_file", None)
if DATA_XLSX is None or not os.path.exists(DATA_XLSX):
    DATA_XLSX = ym_gpr_bundle.get("metadata", {}).get("data_file", None)
if DATA_XLSX is None or not os.path.exists(DATA_XLSX):
    raise FileNotFoundError(
        "Could not find training Excel file via model metadata['data_file'].\n"
        "Manually set DATA_XLSX to your training dataset path."
    )

df_train = safe_read_excel_via_temp_copy(DATA_XLSX)
df_train[FAMILY_NAME_COL] = df_train[FAMILY_NAME_COL].astype(str).str.strip()


# ============================================================
# 8) BOUNDS
# ============================================================

family_bounds = {}
family_bounds_source = {}

for fam, _ in TARGETS_SS:
    b_raw, src = get_family_bounds(df_train, fam, qlo=0.01, qhi=0.99, min_rows=10)
    b = apply_hard_limits(b_raw)
    family_bounds[fam] = b
    family_bounds_source[fam] = src + " + HARD clamp UC[2,4] & t[0.30,0.40]"

print("Bounds used for inverse design (continuous; hard-clamped):")
for fam, _ in TARGETS_SS:
    b = family_bounds[fam]
    print(f" - {fam:8s}: UCx{b['UCx']}, UCy{b['UCy']}, UCz{b['UCz']}, t{b['t']}  | ({family_bounds_source[fam]})")


# ============================================================
# 9) RUN INVERSE DESIGN (GPR + STACKED) - objective = SS target
# ============================================================

all_results_list = []
stage_reports = []

for fam, tgt_ss in TARGETS_SS:
    bounds = family_bounds[fam]
    bsrc = family_bounds_source[fam]

    # GPR
    top_gpr, log_gpr, total_gpr = inverse_design_lhs_continuous_ss(
        family=fam, target_ss=tgt_ss,
        ym_model=YM_GPR, ss_model=SS_GPR,
        bounds=bounds,
        model_name="GPR",
        bounds_source=bsrc
    )
    top_gpr["Total_Evaluated"] = int(total_gpr)
    all_results_list.append(top_gpr)
    stage_reports.append(log_gpr)

    # Stacked
    top_st, log_st, total_st = inverse_design_lhs_continuous_ss(
        family=fam, target_ss=tgt_ss,
        ym_model=YM_STACK, ss_model=SS_STACK,
        bounds=bounds,
        model_name="Stacked Ensemble",
        bounds_source=bsrc
    )
    top_st["Total_Evaluated"] = int(total_st)
    all_results_list.append(top_st)
    stage_reports.append(log_st)

all_results = pd.concat(all_results_list, ignore_index=True)
stage_report = pd.concat(stage_reports, ignore_index=True)


# ============================================================
# 10) ADD ROUNDED PREDICTIONS FOR ALL RESULTS
# ============================================================

def _add_rounded_block(df_block):
    fam = df_block["Family"].iloc[0]
    tgt_ss = float(df_block["Target_SS"].iloc[0])
    model_name = df_block["Model"].iloc[0]
    if model_name == "GPR":
        ym_m, ss_m = YM_GPR, SS_GPR
    else:
        ym_m, ss_m = YM_STACK, SS_STACK
    return add_rounded_predictions_ss(df_block, fam, tgt_ss, ym_m, ss_m)

all_results = (
    all_results.groupby(["Family","Model","Target_SS"], group_keys=False)
    .apply(_add_rounded_block)
    .reset_index(drop=True)
)


# ============================================================
# 11) BEST PICKS (rank by continuous SS error)
# ============================================================

best_overall_family = (
    all_results.sort_values(["Family","SS_error_%","abs_err_ss"])
    .groupby(["Family"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_overall_family = best_overall_family[[
    "Model","Total_Evaluated","Family","Bounds_Source","Target_SS",
    "SS_pred","YM_pred","abs_err_ss","SS_error_%",
    "UCx","UCy","UCz","t",
    "UCx_rounded","UCy_rounded","UCz_rounded","t_rounded",
    "SS_pred_rounded","YM_pred_rounded","abs_err_ss_rounded","SS_error_%_rounded"
]]


# ============================================================
# 12) DISPLAY
# ============================================================

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)

print("\n" + "="*100)
print("BEST OVERALL PER FAMILY (optimize SS; show YM + rounded metrics)")
print("="*100)
display(best_overall_family)


# ============================================================
# 13) SAVE EXCEL + CSV
#     Timestamped folder: TPMS_InverseDesign_TargetSS_YYYYMMDD_HHMMSS
# ============================================================

BASE_OUT = os.path.join(HERE, "outputs")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

out_dir = ensure_dir(os.path.join(BASE_OUT, f"TPMS_InverseDesign_TargetSS_{ts}"))

xlsx_path = os.path.join(out_dir, f"InverseDesign_TargetSS_{ts}.xlsx")
csv_path  = os.path.join(out_dir, f"InverseDesign_TargetSS_{ts}.csv")

# Save CSV first (now folder is guaranteed)
all_results.to_csv(csv_path, index=False)

# Save Excel
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    all_results.to_excel(writer, index=False, sheet_name="ALL_RESULTS")
    best_overall_family.to_excel(writer, index=False, sheet_name="BEST_OVERALL_FAM")
    stage_report.to_excel(writer, index=False, sheet_name="STAGEWISE_REPORT")

    # Family/Model sheets (top 30 per model)
    for fam, _ in TARGETS_SS:
        fam_can = canon_family(fam)
        for model in ["GPR", "Stacked Ensemble"]:
            sh = f"{fam_can}_{model}".replace(" ", "")
            sub = all_results[(all_results["Family"] == fam_can) & (all_results["Model"] == model)].copy()
            sub = sub.sort_values(["SS_error_%","abs_err_ss"]).head(30)
            sub.to_excel(writer, index=False, sheet_name=sh[:31])

print("\nSaved files:")
print(" -", xlsx_path)
print(" -", csv_path)
print("Folder:")
print(" -", out_dir)
