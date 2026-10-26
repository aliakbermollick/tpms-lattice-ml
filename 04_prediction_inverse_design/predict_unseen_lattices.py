"""Predict E and E_specific of the 5 unseen lattices with all 7 models (manuscript Sec. 2.6/3.4, Tables 4-5).
Extracted from prediction 01.03.26.ipynb (cell 0); only paths changed.
Needs scikit-learn==1.5.2 to load the saved models. Set TPMS_MODEL_RUN to the folder
TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46 (default: ../03_ml_models/outputs/<that name>).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_RUN = os.environ.get("TPMS_MODEL_RUN", os.path.join(HERE, "..", "03_ml_models", "outputs", "TPMS_ML_6Models_PLUS_Stack_run_2026-01-08_11-57-46"))
# ============================================================
# TPMS: predict E and E_specific of unseen lattices with the saved models
# (1) ML prediction tables
# (2) Percent error relative to the FE simulation (nTop)
#
# Output folder: TPMS_Predictions_NewModels_20260108
# ============================================================

import os
import re
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime

# -----------------------------
# 1) MODEL PATHS
# -----------------------------
YM_MODEL_PATHS = [
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_ANN_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_Decision_Tree_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_DNN_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_GPR_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_Gradient_Boosting_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_Random_Forest_Youngs_Modulus_MPa.pkl"),
os.path.join(MODEL_RUN, r"Youngs_Modulus_MPa\models\model_StackedEnsemble_Youngs_Modulus_MPa.pkl"),
]

SS_MODEL_PATHS = [
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_ANN_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_Decision_Tree_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_DNN_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_GPR_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_Gradient_Boosting_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_Random_Forest_Specific_Stiffness_MPa-kg-m3.pkl"),
os.path.join(MODEL_RUN, r"Specific_Stiffness_MPa-kg-m3\models\model_StackedEnsemble_Specific_Stiffness_MPa-kg-m3.pkl"),
]

# Same order as in training
FAMILY_ORDER = ["Diamond", "Gyroid", "Split P", "Schwarz", "Lidinoid"]

# Display order
DISPLAY_MODEL_ORDER = ["Decision Tree","Random Forest","Gradient Boosting","GPR","ANN","DNN","Stacked Ensemble"]

# -----------------------------
# 2) STRUCTURES (UCx, UCy, UCz, t)
# -----------------------------
structures = [
    ("Lidinoid (2,2.1,3.5,0.4)", "Lidinoid", 2.0, 2.1, 3.5, 0.4),
    ("Schwarz  (2.2,2.4,2.6,0.4)", "Schwarz", 2.2, 2.4, 2.6, 0.4),
    ("Gyroid (2.5,3.2,3.5,0.3)", "Gyroid", 2.5, 3.2, 3.5, 0.3),
    ("Diamond (2.1,2.2,3.3,0.3)", "Diamond", 2.1, 2.2, 3.3, 0.3),
    ("Split P (3.3,3.4 3.9, 0.3)", "Split P", 3.3, 3.4, 3.9, 0.3),
]

# -----------------------------
# 3) SIMULATION (nTop) VALUES
# -----------------------------
SIM_YM = {
    "Lidinoid (2,2.1,3.5,0.4)": 574.66,
    "Schwarz  (2.2,2.4,2.6,0.4)": 596.29,
    "Gyroid (2.5,3.2,3.5,0.3)": 126.95,
    "Diamond (2.1,2.2,3.3,0.3)": 278.92,
    "Split P (3.3,3.4 3.9, 0.3)": 154.82,
}

SIM_SS = {
    "Lidinoid (2,2.1,3.5,0.4)": 1.046966976,
    "Schwarz  (2.2,2.4,2.6,0.4)": 0.969312785,
    "Gyroid (2.5,3.2,3.5,0.3)": 0.619505783,
    "Diamond (2.1,2.2,3.3,0.3)": 0.93175421,
    "Split P (3.3,3.4 3.9, 0.3)": 0.651670888,
}

# -----------------------------
# 4) HELPERS
# -----------------------------
def load_bundle(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model not found:\n{pkl_path}")
    b = load(pkl_path)
    if not isinstance(b, dict) or "pipeline" not in b:
        raise ValueError(f"Unexpected bundle format in:\n{pkl_path}\nExpected dict with key 'pipeline'.")
    return b

def one_hot_family(family: str, order=FAMILY_ORDER):
    if family not in order:
        raise ValueError(f"Family '{family}' not in FAMILY_ORDER = {order}")
    v = np.zeros(len(order), dtype=float)
    v[order.index(family)] = 1.0
    return v

def make_feature_row(family: str, UCx: float, UCy: float, UCz: float, t: float):
    # IMPORTANT numeric order: (t, UCx, UCy, UCz)
    return np.hstack([one_hot_family(family), np.array([t, UCx, UCy, UCz], dtype=float)])

def extract_model_name_from_filename(p: str):
    base = os.path.basename(p).replace(".pkl", "")
    base = base.replace("model_", "")
    base = re.sub(r"_Youngs_Modulus_MPa$", "", base)
    base = re.sub(r"_Specific_Stiffness_MPa-kg-m3$", "", base)
    base = base.replace("_", " ").strip()
    base = base.replace("StackedEnsemble", "Stacked Ensemble")
    return base

def percent_error(sim, pred):
    sim = float(sim)
    pred = float(pred)
    if sim == 0:
        return np.nan
    return abs(pred - sim) / abs(sim) * 100.0

def make_prediction_table(model_paths, X, index_labels):
    preds = {}
    for p in model_paths:
        b = load_bundle(p)
        name = extract_model_name_from_filename(p)
        y = b["pipeline"].predict(X).astype(float).ravel()
        preds[name] = y

    dfp = pd.DataFrame(preds, index=index_labels)

    ordered = [c for c in DISPLAY_MODEL_ORDER if c in dfp.columns] + \
              [c for c in dfp.columns if c not in DISPLAY_MODEL_ORDER]
    return dfp[ordered]

def make_error_table(sim_map, pred_df):
    err = pred_df.copy()
    for idx in pred_df.index:
        sim_val = sim_map[idx]
        err.loc[idx, :] = [percent_error(sim_val, pred_df.loc[idx, c]) for c in pred_df.columns]
    return err.astype(float)

# -----------------------------
# 5) BUILD FEATURE MATRIX
# -----------------------------
row_labels = [s[0] for s in structures]
X = np.vstack([make_feature_row(fam, UCx, UCy, UCz, t) for (label, fam, UCx, UCy, UCz, t) in structures])

# -----------------------------
# 6) PREDICT YM + SS
# -----------------------------
ym_pred = make_prediction_table(YM_MODEL_PATHS, X, row_labels)
ss_pred = make_prediction_table(SS_MODEL_PATHS, X, row_labels)

# -----------------------------
# 7) ERROR TABLES vs SIMULATION
# -----------------------------
ym_err = make_error_table(SIM_YM, ym_pred)
ss_err = make_error_table(SIM_SS, ss_pred)

# -----------------------------
# 8) COMBINED TABLES (SIM + ML + ERROR)
# -----------------------------
ym_table = pd.DataFrame(index=row_labels)
ym_table[("Young's Modulus (MPa)", "Simulation (nTop)")] = [SIM_YM[i] for i in row_labels]
for c in ym_pred.columns:
    ym_table[("Young's Modulus (MPa)", c)] = ym_pred[c].values
for c in ym_err.columns:
    ym_table[("Error (%)", c)] = ym_err[c].values

ss_table = pd.DataFrame(index=row_labels)
ss_table[("Specific Stiffness (MPa/(kg/m³))", "Simulation (nTop)")] = [SIM_SS[i] for i in row_labels]
for c in ss_pred.columns:
    ss_table[("Specific Stiffness (MPa/(kg/m³))", c)] = ss_pred[c].values
for c in ss_err.columns:
    ss_table[("Error (%)", c)] = ss_err[c].values

ym_table.columns = pd.MultiIndex.from_tuples(ym_table.columns)
ss_table.columns = pd.MultiIndex.from_tuples(ss_table.columns)

# -----------------------------
# 9) DISPLAY
# -----------------------------
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

print("=== Young's Modulus (MPa) Table ===")
display(ym_table)

print("\n=== Specific Stiffness (MPa/(kg/m³)) Table ===")
display(ss_table)

# -----------------------------
# 10) EXPORT (Excel + CSVs)
# -----------------------------
BASE_OUT = os.path.join(HERE, "outputs")
out_dir = os.path.join(BASE_OUT, "TPMS_Predictions_NewModels_20260108")
os.makedirs(out_dir, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_path = os.path.join(out_dir, f"Predictions_Table_YM_SS_{ts}.xlsx")
ym_csv    = os.path.join(out_dir, f"YM_Table_{ts}.csv")
ss_csv    = os.path.join(out_dir, f"SS_Table_{ts}.csv")

# CSV: flatten multiindex for clean CSV headers
ym_flat = ym_table.copy()
ym_flat.columns = [f"{a} | {b}" for a, b in ym_flat.columns]
ym_flat.to_csv(ym_csv, index=True)

ss_flat = ss_table.copy()
ss_flat.columns = [f"{a} | {b}" for a, b in ss_flat.columns]
ss_flat.to_csv(ss_csv, index=True)

# Excel: keep multiindex
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    ym_table.to_excel(writer, sheet_name="Youngs_Modulus_Table")
    ss_table.to_excel(writer, sheet_name="Specific_Stiffness_Table")
    ym_pred.to_excel(writer, sheet_name="YM_ML_only")
    ym_err.to_excel(writer, sheet_name="YM_Error_only(%)")
    ss_pred.to_excel(writer, sheet_name="SS_ML_only")
    ss_err.to_excel(writer, sheet_name="SS_Error_only(%)")

print("\nSaved:")
print(" -", xlsx_path)
print(" -", ym_csv)
print(" -", ss_csv)
print("Output folder:")
print(" -", out_dir)
