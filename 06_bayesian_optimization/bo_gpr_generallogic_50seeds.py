"""GPR with the SE code's generalized prediction logic, 50 seeds. Original run: BO_ContLHS_50seeds_CPU_GPR_generalLogic_2026-03-06_05-43-23 (88.83 min).
Extracted from BO 2.116 xeon date 06.03.24.ipynb (cell 8); only paths, worker count and font changed.
Env vars: MODEL_RUN (folder TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46), OUT_ROOT, N_JOBS (default 40).
"""
import os
from IPython.display import display, Markdown
MODEL_RUN = os.environ["MODEL_RUN"]
# ================================================================
# Parallel Bayesian optimization (50 seeds) with GPR surrogates, using the same
# generalized prediction logic as the SE script
#
# Purpose: timing comparison with the SE version. If the runtime is close to that of
# the original GPR script, the SE model itself is the heavier one.
# ================================================================

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from joblib import load, Parallel, delayed
from skopt import Optimizer
from skopt.space import Real, Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ================================================================
# 0) FORCE ARIAL + EDITABLE PDF/SVG TEXT
# ================================================================
arial_name = "DejaVu Sans"  # Arial not installed

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [arial_name],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

print("Forced font:", arial_name)

# ================================================================
# 1) PATHS (GPR MODELS)
# ================================================================
OUT_ROOT = os.environ.get("OUT_ROOT", ".")

GPR_YM = os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa/models/model_GPR_Youngs_Modulus_MPa.pkl")

GPR_SS = os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3/models/model_GPR_Specific_Stiffness_MPa-kg-m3.pkl")

if not os.path.exists(GPR_YM):
    raise FileNotFoundError(GPR_YM)
if not os.path.exists(GPR_SS):
    raise FileNotFoundError(GPR_SS)

# ================================================================
# 2) CONFIG
# ================================================================
FAMILIES = ["Diamond", "Gyroid", "Split P", "Schwarz", "Lidinoid"]
ENC_NAMES = [f"enc_{f}" for f in FAMILIES]
FAMILY_TAG = {f: f.replace(" ", "_") for f in FAMILIES}

SEEDS = list(range(101, 151))
N_JOBS = int(os.environ.get("N_JOBS", "40"))
BACKEND = "loky"

N_TOTAL = 100
N_INIT  = max(10, int(0.1 * N_TOTAL))
N_EI    = N_TOTAL - N_INIT

UC_LOW, UC_HIGH = 2.0, 4.0
T_LOW,  T_HIGH  = 0.30, 0.40

UC_STEP = 0.01
T_STEP  = 0.01

ACQ_FUNC = "EI"
SS_LABEL = "Pred_SS_MPa_per_kgm3"

MAX_DUP_REASK = 20
P_THRESHOLDS = [0.90, 0.95]

# ================================================================
# 3) ROUNDING / SNAPPING
# ================================================================
def round_to_step(x, step, low=None, high=None, decimals=2):
    x = float(x)
    step = float(step)
    xr = round(x / step) * step
    if low is not None:
        xr = max(float(low), xr)
    if high is not None:
        xr = min(float(high), xr)
    return round(xr, decimals)

def snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw):
    ucx_used = round_to_step(ucx_raw, UC_STEP, UC_LOW, UC_HIGH, decimals=2)
    ucy_used = round_to_step(ucy_raw, UC_STEP, UC_LOW, UC_HIGH, decimals=2)
    ucz_used = round_to_step(ucz_raw, UC_STEP, UC_LOW, UC_HIGH, decimals=2)
    t_used   = round_to_step(t_raw,   T_STEP,  T_LOW,  T_HIGH,  decimals=2)
    return ucx_used, ucy_used, ucz_used, t_used

# ================================================================
# 4) MODEL HELPERS (SAME GENERALIZED LOGIC AS SE VERSION)
# ================================================================
def _one_hot_family(family: str) -> dict:
    enc = {name: 0 for name in ENC_NAMES}
    enc_name = f"enc_{family}"
    if enc_name not in enc:
        raise ValueError(f"Unknown family '{family}'")
    enc[enc_name] = 1
    return enc

def _get_feature_names(bundle):
    if isinstance(bundle, dict):
        if "metadata" in bundle and isinstance(bundle["metadata"], dict):
            meta = bundle["metadata"]
            if "feature_names" in meta:
                return meta["feature_names"]
        if "feature_names" in bundle:
            return bundle["feature_names"]
    raise KeyError("Could not find feature_names in the model bundle.")

def _assemble_row(bundle, family, ucx, ucy, ucz, t):
    feature_names = _get_feature_names(bundle)
    enc = _one_hot_family(family)

    numerics = {
        "Thickness (t) (mm)": float(t),
        "UCx (mm)": float(ucx),
        "UCy (mm)": float(ucy),
        "UCz (mm)": float(ucz),
    }

    row = []
    for f in feature_names:
        if f in enc:
            row.append(enc[f])
        elif f in numerics:
            row.append(numerics[f])
        else:
            raise ValueError(f"Model expects feature '{f}', not provided.")
    return np.array(row, dtype=float).reshape(1, -1)

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

    for key, val in bundle.items():
        if hasattr(val, "predict"):
            return val

    raise KeyError("Could not find a predictor with .predict() inside the model bundle.")

def _predict_bundle(bundle, X_row):
    predictor = _get_predictor_from_bundle(bundle)

    try:
        y = predictor.predict(X_row)
        return float(np.asarray(y).ravel()[0])
    except Exception:
        pass

    if hasattr(predictor, "steps") and len(predictor.steps) >= 1:
        try:
            Xr = predictor[:-1].transform(X_row)
            final_est = predictor.steps[-1][1]
            y = final_est.predict(Xr)
            return float(np.asarray(y).ravel()[0])
        except Exception:
            pass

    raise RuntimeError("Prediction failed for the loaded model bundle.")

# ================================================================
# 5) PROCESS-LOCAL MODEL CACHE
# ================================================================
_MODEL_CACHE = {}

def get_models():
    global _MODEL_CACHE
    if "bYM" not in _MODEL_CACHE:
        _MODEL_CACHE["bYM"] = load(GPR_YM)
    if "bSS" not in _MODEL_CACHE:
        _MODEL_CACHE["bSS"] = load(GPR_SS)
    return _MODEL_CACHE["bYM"], _MODEL_CACHE["bSS"]

def evaluate_surrogates(family, ucx, ucy, ucz, t_used):
    bYM, bSS = get_models()

    X_row_YM = _assemble_row(bYM, family, ucx, ucy, ucz, t_used)
    ym = _predict_bundle(bYM, X_row_YM)

    X_row_SS = _assemble_row(bSS, family, ucx, ucy, ucz, t_used)
    ss = _predict_bundle(bSS, X_row_SS)

    return ym, ss

# ================================================================
# 6) LHS
# ================================================================
def lhs_nd(n, dims, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    result = np.zeros((n, dims), dtype=float)
    for d in range(dims):
        cut = np.linspace(0, 1, n + 1)
        u = rng.rand(n) * (cut[1:] - cut[:-1]) + cut[:-1]
        rng.shuffle(u)
        result[:, d] = u
    return result

def lhs_uc_t(n, rng=None):
    u = lhs_nd(n, dims=4, rng=rng)
    uc = UC_LOW + u[:, :3] * (UC_HIGH - UC_LOW)
    t  = T_LOW  + u[:, 3]  * (T_HIGH  - T_LOW)
    return uc, t

def balanced_families(n_init, families, rng):
    m = len(families)
    base = n_init // m
    rem  = n_init % m
    fams = []
    for f in families:
        fams.extend([f] * base)
    if rem > 0:
        extras = rng.choice(families, size=min(rem, m), replace=False)
        fams.extend(list(extras))
    rng.shuffle(fams)
    return fams[:n_init]

# ================================================================
# 7) FIGURE HELPERS
# ================================================================
def _compute_P(best_so_far):
    best_so_far = np.asarray(best_so_far, dtype=float)
    ymax = float(best_so_far.max())
    return best_so_far / ymax if ymax != 0 else best_so_far

def save_all_formats(fig, out_base, dpi=300):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")

def fig_A_objective(df, out_base, ylabel="Objective"):
    it = df["iter"].values
    y  = df["Objective_value"].values
    best = df["best_so_far"].values
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(it, y, s=16, alpha=0.6)
    ax.plot(it, best, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_all_formats(fig, out_base, dpi=300)
    plt.close(fig)

def fig_B_performance(df, out_base):
    it = df["iter"].values
    P = _compute_P(df["best_so_far"].values)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(it, P, marker="o", ms=3, lw=1.5)
    ax.axhline(1.0, ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_all_formats(fig, out_base, dpi=300)
    plt.close(fig)

def fig_C_distributions(df, out_base):
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.8))
    fam_counts = df["family"].value_counts().reindex(FAMILIES).fillna(0).values
    axes[0].bar(range(len(FAMILIES)), fam_counts)
    axes[0].set_xticks(range(len(FAMILIES)))
    axes[0].set_xticklabels([f.replace(" ", "\n") for f in FAMILIES], rotation=90)
    axes[0].set_ylabel("Count")

    def _hist(ax, col, xlabel, bins=15):
        ax.hist(df[col].values, bins=bins, edgecolor="black", alpha=0.75)
        ax.set_xlabel(xlabel)

    _hist(axes[1], "t_used (mm)", "t (mm)", bins=15)
    _hist(axes[2], "UCx_used (mm)", r"UC$_x$ (mm)", bins=15)
    _hist(axes[3], "UCy_used (mm)", r"UC$_y$ (mm)", bins=15)
    _hist(axes[4], "UCz_used (mm)", r"UC$_z$ (mm)", bins=15)

    for ax in axes:
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_all_formats(fig, out_base, dpi=300)
    plt.close(fig)

def fig_D_trajectory(df, out_base):
    it = df["iter"].values
    fig, axes = plt.subplots(5, 1, figsize=(6.0, 7.5), sharex=True)
    fam_to_y = {f: i for i, f in enumerate(FAMILIES)}
    fam_y = np.array([fam_to_y[f] for f in df["family"].values])

    axes[0].scatter(it, fam_y, s=14, alpha=0.6)
    axes[0].set_yticks(range(len(FAMILIES)))
    axes[0].set_yticklabels(FAMILIES)
    axes[0].set_ylabel("Family")

    axes[1].scatter(it, df["t_used (mm)"].values, s=14, alpha=0.6)
    axes[1].set_ylabel("t (mm)")

    axes[2].scatter(it, df["UCx_used (mm)"].values, s=14, alpha=0.6)
    axes[2].set_ylabel(r"UC$_x$ (mm)")

    axes[3].scatter(it, df["UCy_used (mm)"].values, s=14, alpha=0.6)
    axes[3].set_ylabel(r"UC$_y$ (mm)")

    axes[4].scatter(it, df["UCz_used (mm)"].values, s=14, alpha=0.6)
    axes[4].set_ylabel(r"UC$_z$ (mm)")
    axes[4].set_xlabel("Iteration")

    for ax in axes:
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_all_formats(fig, out_base, dpi=300)
    plt.close(fig)

def fig_E_multi_campaign_P(P_matrix, out_base):
    P = np.asarray(P_matrix, dtype=float)
    it = np.arange(1, P.shape[1] + 1)

    med = np.median(P, axis=0)
    q25 = np.percentile(P, 25, axis=0)
    q75 = np.percentile(P, 75, axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(it, med, lw=2, marker="o", ms=3)
    ax.fill_between(it, q25, q75, alpha=0.25)

    for thr in P_THRESHOLDS:
        ax.axhline(thr, ls="--", lw=1)
        ax.text(2, thr + 0.01, f"P = {thr:.2f}")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_all_formats(fig, out_base, dpi=300)
    plt.close(fig)

# ================================================================
# 8) BO CORE - PER FAMILY
# ================================================================
def run_bo_per_family_worker(family, seed, perfamily_root, objective="YM"):
    rng = np.random.RandomState(seed)
    fam_tag = FAMILY_TAG[family]
    seed_dir = os.path.join(perfamily_root, f"{fam_tag}_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    space = [
        Real(UC_LOW, UC_HIGH, name="UCx"),
        Real(UC_LOW, UC_HIGH, name="UCy"),
        Real(UC_LOW, UC_HIGH, name="UCz"),
        Real(T_LOW,  T_HIGH,  name="t"),
    ]
    opt = Optimizer(space, base_estimator="GP", acq_func=ACQ_FUNC, random_state=seed)

    evaluated_used = set()
    rows = []
    lhs_uc, lhs_t = lhs_uc_t(N_INIT, rng=rng)

    for i in range(N_INIT):
        ucx_raw, ucy_raw, ucz_raw = lhs_uc[i]
        t_raw = float(lhs_t[i])
        ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)
        key = (ucx_used, ucy_used, ucz_used, t_used)

        ym, ss = evaluate_surrogates(family, ucx_used, ucy_used, ucz_used, t_used)
        obj_val = ym if objective == "YM" else ss

        opt.tell([float(ucx_raw), float(ucy_raw), float(ucz_raw), float(t_raw)], -obj_val)
        evaluated_used.add(key)

        rows.append({
            "iter": i + 1, "phase": "LHS", "family": family, "seed": seed, "objective": objective,
            "UCx_raw (mm)": float(ucx_raw), "UCy_raw (mm)": float(ucy_raw),
            "UCz_raw (mm)": float(ucz_raw), "t_raw (mm)": float(t_raw),
            "UCx_used (mm)": ucx_used, "UCy_used (mm)": ucy_used,
            "UCz_used (mm)": ucz_used, "t_used (mm)": t_used,
            "Pred_YM_MPa": ym, SS_LABEL: ss, "Objective_value": obj_val
        })

    for k in range(N_EI):
        x = opt.ask()
        for _ in range(MAX_DUP_REASK):
            ucx_raw, ucy_raw, ucz_raw, t_raw = map(float, x)
            ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)
            key = (ucx_used, ucy_used, ucz_used, t_used)
            if key not in evaluated_used:
                break
            x = opt.ask()

        ucx_raw, ucy_raw, ucz_raw, t_raw = map(float, x)
        ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)

        ym, ss = evaluate_surrogates(family, ucx_used, ucy_used, ucz_used, t_used)
        obj_val = ym if objective == "YM" else ss

        opt.tell([ucx_raw, ucy_raw, ucz_raw, t_raw], -obj_val)
        evaluated_used.add((ucx_used, ucy_used, ucz_used, t_used))

        rows.append({
            "iter": N_INIT + k + 1, "phase": "EI", "family": family, "seed": seed, "objective": objective,
            "UCx_raw (mm)": ucx_raw, "UCy_raw (mm)": ucy_raw, "UCz_raw (mm)": ucz_raw, "t_raw (mm)": t_raw,
            "UCx_used (mm)": ucx_used, "UCy_used (mm)": ucy_used, "UCz_used (mm)": ucz_used, "t_used (mm)": t_used,
            "Pred_YM_MPa": ym, SS_LABEL: ss, "Objective_value": obj_val
        })

    df_log = pd.DataFrame(rows)
    df_log["best_so_far"] = df_log["Objective_value"].cummax()
    df_log.to_csv(os.path.join(seed_dir, "bo_log.csv"), index=False)

    best_obj = df_log.loc[df_log["Objective_value"].idxmax()].copy()
    pd.DataFrame([{
        "mode": "PerFamily", "objective": objective, "family_fixed": family, "seed": seed,
        "Best_Objective": float(best_obj["Objective_value"]),
        "Pred_YM_MPa": float(best_obj["Pred_YM_MPa"]),
        SS_LABEL: float(best_obj[SS_LABEL]),
        "UCx_used (mm)": best_obj["UCx_used (mm)"],
        "UCy_used (mm)": best_obj["UCy_used (mm)"],
        "UCz_used (mm)": best_obj["UCz_used (mm)"],
        "t_used (mm)": best_obj["t_used (mm)"],
    }]).to_csv(os.path.join(seed_dir, "bo_best_summary.csv"), index=False)

    fig_dir = os.path.join(seed_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    obj_label = "E (MPa)" if objective == "YM" else "Specific Stiffness"
    fig_A_objective(df_log, os.path.join(fig_dir, f"A_Obj_seed_{seed}"), ylabel=obj_label)
    fig_B_performance(df_log, os.path.join(fig_dir, f"B_P_seed_{seed}"))
    fig_C_distributions(df_log, os.path.join(fig_dir, f"C_Dist_seed_{seed}"))
    fig_D_trajectory(df_log, os.path.join(fig_dir, f"D_Traj_seed_{seed}"))

    return pd.DataFrame([{
        "mode": "PerFamily", "objective": objective, "family_fixed": family, "seed": seed,
        "Best_Objective": float(best_obj["Objective_value"]),
        "Pred_YM_MPa": float(best_obj["Pred_YM_MPa"]),
        SS_LABEL: float(best_obj[SS_LABEL]),
        "UCx_used (mm)": best_obj["UCx_used (mm)"],
        "UCy_used (mm)": best_obj["UCy_used (mm)"],
        "UCz_used (mm)": best_obj["UCz_used (mm)"],
        "t_used (mm)": best_obj["t_used (mm)"],
    }]), _compute_P(df_log["best_so_far"].values)

# ================================================================
# 9) BO CORE - MIXED
# ================================================================
def run_bo_mixed_worker(seed, mixed_root, objective="YM"):
    rng = np.random.RandomState(seed)
    seed_dir = os.path.join(mixed_root, f"Mixed_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    space = [
        Categorical(FAMILIES, name="family"),
        Real(UC_LOW, UC_HIGH, name="UCx"),
        Real(UC_LOW, UC_HIGH, name="UCy"),
        Real(UC_LOW, UC_HIGH, name="UCz"),
        Real(T_LOW,  T_HIGH,  name="t"),
    ]
    opt = Optimizer(space, base_estimator="GP", acq_func=ACQ_FUNC, random_state=seed)

    evaluated_used = set()
    rows = []
    lhs_uc, lhs_t = lhs_uc_t(N_INIT, rng=rng)
    fam_init = balanced_families(N_INIT, FAMILIES, rng)

    for i in range(N_INIT):
        fam = str(fam_init[i])
        ucx_raw, ucy_raw, ucz_raw = lhs_uc[i]
        t_raw = float(lhs_t[i])
        ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)
        key = (fam, ucx_used, ucy_used, ucz_used, t_used)

        ym, ss = evaluate_surrogates(fam, ucx_used, ucy_used, ucz_used, t_used)
        obj_val = ym if objective == "YM" else ss

        opt.tell([fam, float(ucx_raw), float(ucy_raw), float(ucz_raw), float(t_raw)], -obj_val)
        evaluated_used.add(key)

        rows.append({
            "iter": i + 1, "phase": "LHS_balanced", "family": fam, "seed": seed, "objective": objective,
            "UCx_raw (mm)": float(ucx_raw), "UCy_raw (mm)": float(ucy_raw),
            "UCz_raw (mm)": float(ucz_raw), "t_raw (mm)": float(t_raw),
            "UCx_used (mm)": ucx_used, "UCy_used (mm)": ucy_used,
            "UCz_used (mm)": ucz_used, "t_used (mm)": t_used,
            "Pred_YM_MPa": ym, SS_LABEL: ss, "Objective_value": obj_val
        })

    for k in range(N_EI):
        x = opt.ask()
        for _ in range(MAX_DUP_REASK):
            fam, ucx_raw, ucy_raw, ucz_raw, t_raw = x
            fam = str(fam)
            ucx_raw, ucy_raw, ucz_raw, t_raw = float(ucx_raw), float(ucy_raw), float(ucz_raw), float(t_raw)
            ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)
            key = (fam, ucx_used, ucy_used, ucz_used, t_used)
            if key not in evaluated_used:
                break
            x = opt.ask()

        fam, ucx_raw, ucy_raw, ucz_raw, t_raw = x
        fam = str(fam)
        ucx_raw, ucy_raw, ucz_raw, t_raw = float(ucx_raw), float(ucy_raw), float(ucz_raw), float(t_raw)
        ucx_used, ucy_used, ucz_used, t_used = snap_design(ucx_raw, ucy_raw, ucz_raw, t_raw)

        ym, ss = evaluate_surrogates(fam, ucx_used, ucy_used, ucz_used, t_used)
        obj_val = ym if objective == "YM" else ss

        opt.tell([fam, ucx_raw, ucy_raw, ucz_raw, t_raw], -obj_val)
        evaluated_used.add((fam, ucx_used, ucy_used, ucz_used, t_used))

        rows.append({
            "iter": N_INIT + k + 1, "phase": "EI", "family": fam, "seed": seed, "objective": objective,
            "UCx_raw (mm)": ucx_raw, "UCy_raw (mm)": ucy_raw, "UCz_raw (mm)": ucz_raw, "t_raw (mm)": t_raw,
            "UCx_used (mm)": ucx_used, "UCy_used (mm)": ucy_used, "UCz_used (mm)": ucz_used, "t_used (mm)": t_used,
            "Pred_YM_MPa": ym, SS_LABEL: ss, "Objective_value": obj_val
        })

    df_log = pd.DataFrame(rows)
    df_log["best_so_far"] = df_log["Objective_value"].cummax()
    df_log.to_csv(os.path.join(seed_dir, "bo_log.csv"), index=False)

    best_obj = df_log.loc[df_log["Objective_value"].idxmax()].copy()
    pd.DataFrame([{
        "mode": "Mixed", "objective": objective, "seed": seed, "best_family": best_obj["family"],
        "Best_Objective": float(best_obj["Objective_value"]),
        "Pred_YM_MPa": float(best_obj["Pred_YM_MPa"]),
        SS_LABEL: float(best_obj[SS_LABEL]),
        "UCx_used (mm)": best_obj["UCx_used (mm)"],
        "UCy_used (mm)": best_obj["UCy_used (mm)"],
        "UCz_used (mm)": best_obj["UCz_used (mm)"],
        "t_used (mm)": best_obj["t_used (mm)"],
    }]).to_csv(os.path.join(seed_dir, "bo_best_summary.csv"), index=False)

    fig_dir = os.path.join(seed_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    obj_label = "E (MPa)" if objective == "YM" else "Specific Stiffness"
    fig_A_objective(df_log, os.path.join(fig_dir, f"A_Obj_seed_{seed}"), ylabel=obj_label)
    fig_B_performance(df_log, os.path.join(fig_dir, f"B_P_seed_{seed}"))
    fig_C_distributions(df_log, os.path.join(fig_dir, f"C_Dist_seed_{seed}"))
    fig_D_trajectory(df_log, os.path.join(fig_dir, f"D_Traj_seed_{seed}"))

    return pd.DataFrame([{
        "mode": "Mixed", "objective": objective, "seed": seed, "best_family": best_obj["family"],
        "Best_Objective": float(best_obj["Objective_value"]),
        "Pred_YM_MPa": float(best_obj["Pred_YM_MPa"]),
        SS_LABEL: float(best_obj[SS_LABEL]),
        "UCx_used (mm)": best_obj["UCx_used (mm)"],
        "UCy_used (mm)": best_obj["UCy_used (mm)"],
        "UCz_used (mm)": best_obj["UCz_used (mm)"],
        "t_used (mm)": best_obj["t_used (mm)"],
    }]), _compute_P(df_log["best_so_far"].values)

# ================================================================
# 10) CAMPAIGN WRAPPER
# ================================================================
def run_objective_campaign(objective, objective_root):
    mixed_root = os.path.join(objective_root, "Mixed")
    perf_root  = os.path.join(objective_root, "PerFamily")
    sum_dir    = os.path.join(objective_root, "Summary")
    os.makedirs(mixed_root, exist_ok=True)
    os.makedirs(perf_root, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

    print(f"\n================ {objective}: MIXED =================")
    mixed_results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
        delayed(run_bo_mixed_worker)(seed, mixed_root, objective) for seed in SEEDS
    )

    mixed_summaries = [r[0] for r in mixed_results]
    mixed_P = [r[1] for r in mixed_results]
    df_mixed_all = pd.concat(mixed_summaries, ignore_index=True)
    df_mixed_all.to_csv(os.path.join(sum_dir, f"MasterSummary_Mixed_{objective}.csv"), index=False)
    df_mixed_all.sort_values("Best_Objective", ascending=False).head(1).to_csv(
        os.path.join(sum_dir, f"BestOverall_Mixed_{objective}.csv"), index=False
    )
    fig_E_multi_campaign_P(np.vstack(mixed_P), os.path.join(objective_root, f"FigE_Mixed_{objective}"))

    print(f"\n================ {objective}: PERFAMILY =================")
    pf_jobs = [(fam, seed) for fam in FAMILIES for seed in SEEDS]
    pf_results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
        delayed(run_bo_per_family_worker)(fam, seed, os.path.join(perf_root, FAMILY_TAG[fam]), objective)
        for fam, seed in pf_jobs
    )

    pf_summaries = [r[0] for r in pf_results]
    df_pf_all = pd.concat(pf_summaries, ignore_index=True)
    df_pf_all.to_csv(os.path.join(sum_dir, f"MasterSummary_PerFamily_{objective}.csv"), index=False)
    (
        df_pf_all.sort_values("Best_Objective", ascending=False)
                 .groupby("family_fixed", as_index=False)
                 .first()
    ).to_csv(os.path.join(sum_dir, f"BestPerFamily_{objective}.csv"), index=False)

    pf_P_map = {fam: [] for fam in FAMILIES}
    for (fam, seed), result in zip(pf_jobs, pf_results):
        pf_P_map[fam].append(result[1])

    for fam in FAMILIES:
        fig_E_multi_campaign_P(np.vstack(pf_P_map[fam]),
                               os.path.join(objective_root, f"FigE_PerFamily_{FAMILY_TAG[fam]}_{objective}"))

# ================================================================
# 11) RUNTIME
# ================================================================
def format_elapsed_time(seconds):
    seconds = float(seconds)
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    total_seconds_int = int(round(seconds))
    h = total_seconds_int // 3600
    m = (total_seconds_int % 3600) // 60
    s = total_seconds_int % 60
    return {"seconds": seconds, "minutes": minutes, "hours": hours, "hms": f"{h:02d}:{m:02d}:{s:02d}"}

# ================================================================
# 12) MAIN
# ================================================================
overall_t0 = time.time()

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
master_dir = os.path.join(OUT_ROOT, f"BO_ContLHS_50seeds_CPU_GPR_generalLogic_{ts}")
os.makedirs(master_dir, exist_ok=True)

print("\nMaster output folder:")
print(master_dir)

ym_root = os.path.join(master_dir, "YM_Optimized")
ss_root = os.path.join(master_dir, "SS_Optimized")
os.makedirs(ym_root, exist_ok=True)
os.makedirs(ss_root, exist_ok=True)

run_objective_campaign("YM", ym_root)
run_objective_campaign("SS", ss_root)

elapsed = format_elapsed_time(time.time() - overall_t0)

print("\n=== COMPLETE: GPR with generalized prediction logic ===")
print("Master folder:", master_dir)
print(f"HH:MM:SS  : {elapsed['hms']}")
print(f"Seconds   : {elapsed['seconds']:.2f}")
print(f"Minutes   : {elapsed['minutes']:.2f}")
print(f"Hours     : {elapsed['hours']:.2f}")
