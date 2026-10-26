"""Train 6 base models + Stacked Ensemble for E and E_specific (manuscript Sec. 2.5, 3.3; Table 3, Table S1).

Extracted unchanged (except data/output paths) from TPMS 01.03.26.ipynb, cell 0.
Full grid search is slow (hours). Saved models in the original run need scikit-learn==1.5.2 to load.
Verify existing models without retraining: python verify_saved_models.py <run_dir>
"""
# ================================================================
# TPMS (5 families): ML models for Young's modulus and specific stiffness
# Base models: Decision Tree, Random Forest, Gradient Boosting, GPR, ANN, DNN
# Stacked ensemble: Random Forest + Gradient Boosting + GPR + DNN (Ridge meta-learner)
#
# Outputs for every model and target:
#   - parity plot (train/test R2, RMSE, MAE; axes start at zero)
#   - learning curve (training size vs RMSE, family-stratified CV)
#   - residual plot (predicted vs residual)
#
# Implementation:
# - scikit-learn only; StandardScaler (z-score) inside every pipeline
# - 80/20 train/test split stratified by lattice family
# - GridSearchCV with family-stratified k-fold CV
# - the Excel file is read through a temporary copy to avoid OneDrive/Excel locks
# ================================================================

import os, ast, json, time, shutil, tempfile, warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, figures are only saved
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from joblib import dump

warnings.filterwarnings("ignore")

# -------------------- PATHS --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(HERE, "..", "data", "TPMS_data_27.12.25.xlsx")
OUT_BASE_ROOT = os.path.join(HERE, "outputs")

# -------------------- CONFIG --------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_SPLITS     = 5

TARGETS = [
    "Young's Modulus (MPa)",
    "Specific Stiffness (MPa/(kg/m3))"
]

FAMILY_NAME_COL = "Lattice Type"
ENC_COL         = "Structure Encoding"
NUM_COLS        = ["Thickness (t) (mm)", "UCx (mm)", "UCy (mm)", "UCz (mm)"]

# Must match the one-hot order used in the dataset
FAMILY_ORDER = ["Diamond", "Gyroid", "Split P", "Schwarz", "Lidinoid"]

# Learning curve points (8 points from 10% to 100% of training data)
LC_TRAIN_SIZES = np.linspace(0.10, 1.0, 8)

# Stacking: base models to include
STACK_BASE_NAMES = ["Random Forest", "Gradient Boosting", "GPR", "DNN"]

# ----------------- UTILITIES -----------------
def sanitize(s: str) -> str:
    return (str(s).replace(" ", "_").replace("(", "")
                 .replace(")", "").replace("/", "-")
                 .replace("'", "").replace(",", ""))

def ensure_arraylike(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, str):
        try:
            return list(ast.literal_eval(v))
        except Exception:
            return None
    return None

def decode_family_from_onehot(onehot, order=FAMILY_ORDER):
    enc = ensure_arraylike(onehot)
    if enc is None:
        return None
    try:
        idx = int(np.argmax(enc))
        return order[idx] if 0 <= idx < len(order) else None
    except Exception:
        return None

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_read_excel_via_temp_copy(xlsx_path, sheet_name=0):
    """Avoids PermissionError by copying the Excel file to a temp location first."""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    tmp_dir = tempfile.mkdtemp(prefix="tpms_excel_")
    tmp_xlsx = os.path.join(tmp_dir, os.path.basename(xlsx_path))

    try:
        shutil.copy2(xlsx_path, tmp_xlsx)
    except PermissionError as e:
        raise PermissionError(
            "PermissionError while copying the Excel file.\n"
            "Most likely the file is OPEN in Excel or OneDrive is locking it.\n"
            "Close Excel and retry.\n\n"
            f"Original error: {e}"
        )

    try:
        df_local = pd.read_excel(tmp_xlsx, sheet_name=sheet_name, engine="openpyxl")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return df_local

def make_run_dir(base_root, tag="TPMS_ML_6Models_PLUS_Stack"):
    os.makedirs(base_root, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_root, f"{tag}_run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def target_axis_labels(target: str):
    tl = target.lower()
    if "young" in tl and "modulus" in tl:
        prop = "Young's Modulus (MPa)"
    elif "specific stiffness" in tl:
        prop = "Specific Stiffness (MPa/(kg/m³))"
    else:
        prop = target
    return f"True {prop}", f"Predicted {prop}", prop

def parity_plot(
    y_true, y_pred,
    title, save_path,
    x_label, y_label,
    r2_train, rmse_train, mae_train,
    r2_test, rmse_test, mae_test
):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    hi = float(np.max([np.max(y_true), np.max(y_pred)]))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    xmax = hi * 1.05

    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([0, xmax], [0, xmax], 'k--', lw=1)

    plt.xlim(0, xmax)
    plt.ylim(0, xmax)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)

    txt = (
        "Train\n"
        f"  R²   = {r2_train:.3f}\n"
        f"  RMSE = {rmse_train:.3f}\n"
        f"  MAE  = {mae_train:.3f}\n"
        "Test\n"
        f"  R²   = {r2_test:.3f}\n"
        f"  RMSE = {rmse_test:.3f}\n"
        f"  MAE  = {mae_test:.3f}"
    )

    ax = plt.gca()
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def residual_plot(y_true, y_pred, title, save_path, prop_label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    hi = float(np.max(y_pred))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    xmax = hi * 1.05

    plt.figure(figsize=(6.4, 4.9))
    plt.scatter(y_pred, resid, s=18, alpha=0.75)
    plt.axhline(0.0, linestyle="--", linewidth=1, color="k")
    plt.xlim(0, xmax)

    plt.xlabel(f"Predicted {prop_label}")
    plt.ylabel("Residual (True − Predicted)")
    plt.title(title)
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def learning_curve_plot(estimator, X_train, y_train, cv, title, save_path):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=LC_TRAIN_SIZES,
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_rmse = -np.mean(train_scores, axis=1)
    train_std  = np.std(-train_scores, axis=1)

    valid_rmse = -np.mean(valid_scores, axis=1)
    valid_std  = np.std(-valid_scores, axis=1)

    plt.figure(figsize=(6.4, 4.9))
    plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(train_sizes, valid_rmse, marker="o", label="Cross-val RMSE")

    plt.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.15)
    plt.fill_between(train_sizes, valid_rmse - valid_std, valid_rmse + valid_std, alpha=0.15)

    plt.xlabel("Number of training samples")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def timed_fit(gs, X_train, y_train, label=""):
    t0 = time.time()
    gs.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"[{label}] GridSearchCV finished in {dt/60:.1f} min")
    return gs

class FamilyStratifiedKFold(StratifiedKFold):
    """Stratify CV folds using family labels."""
    def __init__(self, labels, n_splits=5, shuffle=True, random_state=42):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._labels = np.asarray(labels)

    def split(self, X, y=None, groups=None):
        labels = self._labels
        if len(labels) != len(X):
            labels = labels[:len(X)]
        return super().split(X, labels, groups)

# ----------------- MODEL SPACES (ALL 6) -----------------
def make_dt():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", DecisionTreeRegressor(random_state=RANDOM_STATE))
    ])
    grid = {
        "regressor__criterion": ["squared_error"],
        "regressor__splitter": ["best", "random"],
        "regressor__max_depth": [None, 5, 7, 10, 15],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 5],
        "regressor__max_features": [None, "sqrt", "log2"]
    }
    return pipe, grid

def make_rf():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    grid = {
        "regressor__n_estimators": [300, 500, 800],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
        "regressor__max_features": [None, "sqrt", "log2"],
        "regressor__bootstrap": [True]
    }
    return pipe, grid

def make_gb():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    grid = {
        "regressor__n_estimators": [400, 800],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [2, 3, 4],
        "regressor__subsample": [0.7, 1.0],
        "regressor__max_features": ["sqrt", None]
    }
    return pipe, grid

def make_gpr(n_features):
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=RANDOM_STATE,
            n_restarts_optimizer=2
        ))
    ])
    grid = {
        "regressor__alpha": [1e-6, 1e-5, 1e-4],
        "regressor__n_restarts_optimizer": [0, 2, 5]
    }
    return pipe, grid

def make_ann():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            random_state=RANDOM_STATE,
            max_iter=900,
            early_stopping=True,
            n_iter_no_change=20
        ))
    ])
    grid = {
        "regressor__hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "regressor__activation": ["relu", "tanh"],
        "regressor__alpha": [1e-5, 1e-4, 1e-3],
        "regressor__learning_rate_init": [1e-3, 5e-4],
        "regressor__batch_size": [16, 32, 64]
    }
    return pipe, grid

def make_dnn():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            random_state=RANDOM_STATE,
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=20
        ))
    ])
    grid = {
        "regressor__hidden_layer_sizes": [(256, 128, 64), (128, 128, 64), (256, 128)],
        "regressor__activation": ["relu"],
        "regressor__alpha": [1e-5, 1e-4, 1e-3],
        "regressor__learning_rate_init": [1e-3, 5e-4, 1e-4],
        "regressor__batch_size": [32, 64, 128]
    }
    return pipe, grid

MODEL_BUILDERS = {
    "Decision Tree": make_dt,
    "Random Forest": make_rf,
    "Gradient Boosting": make_gb,
    "GPR": lambda: make_gpr(n_features=len(feature_names)) if "feature_names" in globals() else None,  # patched after features
    "ANN": make_ann,
    "DNN": make_dnn,
}

# ----------------- LOAD DATA -----------------
print("Using dataset:", DATA_XLSX)
df = safe_read_excel_via_temp_copy(DATA_XLSX)

required_cols = [FAMILY_NAME_COL, ENC_COL] + NUM_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in Excel: {missing}\nFound columns: {list(df.columns)}")

df[ENC_COL] = df[ENC_COL].apply(ensure_arraylike)
if df[ENC_COL].isna().any():
    bad_idx = df[df[ENC_COL].isna()].index[:10].tolist()
    raise ValueError(f"Invalid '{ENC_COL}' values in rows (examples): {bad_idx}")

df[FAMILY_NAME_COL] = df[FAMILY_NAME_COL].astype(str).str.strip()
missing_family = df[FAMILY_NAME_COL].isna() | (df[FAMILY_NAME_COL].str.lower() == "nan") | (df[FAMILY_NAME_COL] == "")
if missing_family.any():
    df.loc[missing_family, FAMILY_NAME_COL] = df.loc[missing_family, ENC_COL].apply(decode_family_from_onehot)

if df[FAMILY_NAME_COL].isna().any():
    raise ValueError("Some family labels are missing; cannot stratify.")

for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=NUM_COLS).reset_index(drop=True)

# Features: one-hot + numeric
ENC_LEN = len(FAMILY_ORDER)

def pad_or_trim(enc, want=ENC_LEN):
    enc = list(enc)
    if len(enc) < want:
        enc += [0] * (want - len(enc))
    if len(enc) > want:
        enc = enc[:want]
    return enc

enc_matrix = np.vstack(df[ENC_COL].apply(pad_or_trim).values).astype(float)
X_all = np.hstack([enc_matrix, df[NUM_COLS].values])

enc_colnames  = [f"enc_{name}" for name in FAMILY_ORDER]
feature_names = enc_colnames + NUM_COLS

# patch GPR builder now that feature_names exists
MODEL_BUILDERS["GPR"] = lambda: make_gpr(n_features=len(feature_names))

y_family_all = df[FAMILY_NAME_COL].astype(str).values

# Single fixed split for fairness
X_train_all, X_test_all, fam_train_all, fam_test_all, idx_tr, idx_te = train_test_split(
    X_all, y_family_all, np.arange(len(df)),
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_family_all
)

# ----------------- STACKING BUILDER (ONLY 4 BASE MODELS) -----------------
def make_stacked(best_estimators_dict):
    """
    best_estimators_dict: dict of model_name -> tuned pipeline (best estimator)
    Uses ONLY STACK_BASE_NAMES from that dict.
    Meta-model: Ridge, tuned over alpha.
    """
    estimators = [(name, best_estimators_dict[name]) for name in STACK_BASE_NAMES]

    stack = Pipeline([
        ("scaler", StandardScaler()),  # optional, keeps meta-level stable; base models already scale internally
        ("regressor", StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=RANDOM_STATE),
            passthrough=False,
            cv=N_SPLITS,
            n_jobs=-1
        ))
    ])

    grid = {
        "regressor__final_estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    return stack, grid

# ----------------- RUN -----------------
RUN_DIR = make_run_dir(OUT_BASE_ROOT)
print("\nOutputs will be saved to:\n", RUN_DIR)

summary_rows = []
overall_t0 = time.time()

for target in TARGETS:
    if target not in df.columns:
        print(f"[WARN] Target '{target}' not found; skipping.")
        continue

    # target values
    y_full = pd.to_numeric(df[target], errors="coerce").values
    ok = ~np.isnan(y_full)
    if ok.sum() < 10:
        print(f"[WARN] Too few valid rows for '{target}'; skipping.")
        continue

    # filter train/test by valid target
    tr_mask = ok[idx_tr]
    te_mask = ok[idx_te]

    X_train = X_train_all[tr_mask]
    X_test  = X_test_all[te_mask]
    y_train = y_full[idx_tr][tr_mask]
    y_test  = y_full[idx_te][te_mask]
    fam_train = fam_train_all[tr_mask]

    # family-stratified CV
    cv_splitter = FamilyStratifiedKFold(labels=fam_train, n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # directories for this target
    TGT_DIR    = os.path.join(RUN_DIR, sanitize(target))
    TGT_TABLES = os.path.join(TGT_DIR, "tables")
    TGT_PLOTS  = os.path.join(TGT_DIR, "plots")
    TGT_MODELS = os.path.join(TGT_DIR, "models")
    for d in [TGT_DIR, TGT_TABLES, TGT_PLOTS, TGT_MODELS]:
        os.makedirs(d, exist_ok=True)

    xlab, ylab, prop_label = target_axis_labels(target)

    # ----- Train & tune ALL 6 models -----
    best_estimators = {}  # model_name -> best estimator (pipeline)

    for model_name in ["Decision Tree","Random Forest","Gradient Boosting","GPR","ANN","DNN"]:
        print(f"\n=== {model_name} → {target} ===")
        pipe, param_grid = MODEL_BUILDERS[model_name]()

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=True
        )
        timed_fit(gs, X_train, y_train, f"{model_name} → {target}")

        best = gs.best_estimator_
        best_params = gs.best_params_
        best_cv_rmse = -float(gs.best_score_)
        best_estimators[model_name] = best

        # Metrics
        y_pred_tr = best.predict(X_train)
        y_pred_te = best.predict(X_test)

        r2_tr   = r2_score(y_train, y_pred_tr)
        rmse_tr = rmse(y_train, y_pred_tr)
        mae_tr  = mean_absolute_error(y_train, y_pred_tr)

        r2_te   = r2_score(y_test, y_pred_te)
        rmse_te = rmse(y_test, y_pred_te)
        mae_te  = mean_absolute_error(y_test, y_pred_te)

        # Plots
        fig_parity = os.path.join(TGT_PLOTS, f"Parity_{sanitize(model_name)}_{sanitize(target)}.png")
        parity_plot(
            y_true=y_test, y_pred=y_pred_te,
            title=f"{model_name} — {target}",
            save_path=fig_parity,
            x_label=xlab, y_label=ylab,
            r2_train=r2_tr, rmse_train=rmse_tr, mae_train=mae_tr,
            r2_test=r2_te, rmse_test=rmse_te, mae_test=mae_te
        )

        fig_resid = os.path.join(TGT_PLOTS, f"Residual_{sanitize(model_name)}_{sanitize(target)}.png")
        residual_plot(
            y_true=y_test, y_pred=y_pred_te,
            title=f"Residuals: {model_name} — {target}",
            save_path=fig_resid,
            prop_label=prop_label
        )

        fig_lc = os.path.join(TGT_PLOTS, f"LearningCurve_{sanitize(model_name)}_{sanitize(target)}.png")
        try:
            learning_curve_plot(
                estimator=best,
                X_train=X_train,
                y_train=y_train,
                cv=cv_splitter,
                title=f"Learning Curve: {model_name} — {target}",
                save_path=fig_lc
            )
        except Exception as e:
            fig_lc = ""
            print(f"[WARN] Learning curve failed for {model_name} → {target}: {e}")

        # Save CV table
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_path = os.path.join(TGT_TABLES, f"CV_{sanitize(model_name)}_{sanitize(target)}.csv")
        cv_df.to_csv(cv_path, index=False)

        # Save model bundle
        mdl_path = os.path.join(TGT_MODELS, f"model_{sanitize(model_name)}_{sanitize(target)}.pkl")
        bundle = {
            "pipeline": best,
            "metadata": {
                "model_name": model_name,
                "target": target,
                "feature_names": feature_names,
                "family_order": FAMILY_ORDER,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "n_splits": N_SPLITS,
                "best_params": best_params,
                "best_cv_rmse": best_cv_rmse,
                "data_file": DATA_XLSX
            }
        }
        # Add GPR kernel string if applicable
        try:
            inner = best.named_steps["regressor"]
            if isinstance(inner, GaussianProcessRegressor):
                bundle["metadata"]["gpr_kernel_"] = str(inner.kernel_)
        except Exception:
            pass

        dump(bundle, mdl_path)

        summary_rows.append({
            "Model": model_name,
            "Target": target,
            "Best CV RMSE": round(best_cv_rmse, 6),
            "Train MAE": round(mae_tr, 6),
            "Train RMSE": round(rmse_tr, 6),
            "Train R²": round(r2_tr, 6),
            "Test MAE": round(mae_te, 6),
            "Test RMSE": round(rmse_te, 6),
            "Test R²": round(r2_te, 6),
            "Best Params": json.dumps(best_params),
            "Model Path": mdl_path,
            "CV Table": cv_path,
            "Parity Plot": fig_parity,
            "Residual Plot": fig_resid,
            "Learning Curve": fig_lc
        })

    # ----- Train & tune STACKED ENSEMBLE using ONLY RF+GB+GPR+DNN -----
    print(f"\n=== Stacked Ensemble (RF+GB+GPR+DNN) → {target} ===")
    stack_pipe, stack_grid = make_stacked(best_estimators)

    stack_gs = GridSearchCV(
        estimator=stack_pipe,
        param_grid=stack_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_splitter,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=True
    )
    timed_fit(stack_gs, X_train, y_train, f"Stacked Ensemble → {target}")

    stack_best = stack_gs.best_estimator_
    stack_best_params = stack_gs.best_params_
    stack_best_cv_rmse = -float(stack_gs.best_score_)

    y_pred_tr = stack_best.predict(X_train)
    y_pred_te = stack_best.predict(X_test)

    r2_tr   = r2_score(y_train, y_pred_tr)
    rmse_tr = rmse(y_train, y_pred_tr)
    mae_tr  = mean_absolute_error(y_train, y_pred_tr)

    r2_te   = r2_score(y_test, y_pred_te)
    rmse_te = rmse(y_test, y_pred_te)
    mae_te  = mean_absolute_error(y_test, y_pred_te)

    fig_parity = os.path.join(TGT_PLOTS, f"Parity_StackedEnsemble_{sanitize(target)}.png")
    parity_plot(
        y_true=y_test, y_pred=y_pred_te,
        title=f"Stacked Ensemble — {target}",
        save_path=fig_parity,
        x_label=xlab, y_label=ylab,
        r2_train=r2_tr, rmse_train=rmse_tr, mae_train=mae_tr,
        r2_test=r2_te, rmse_test=rmse_te, mae_test=mae_te
    )

    fig_resid = os.path.join(TGT_PLOTS, f"Residual_StackedEnsemble_{sanitize(target)}.png")
    residual_plot(
        y_true=y_test, y_pred=y_pred_te,
        title=f"Residuals: Stacked Ensemble — {target}",
        save_path=fig_resid,
        prop_label=prop_label
    )

    fig_lc = os.path.join(TGT_PLOTS, f"LearningCurve_StackedEnsemble_{sanitize(target)}.png")
    try:
        learning_curve_plot(
            estimator=stack_best,
            X_train=X_train,
            y_train=y_train,
            cv=cv_splitter,
            title=f"Learning Curve: Stacked Ensemble — {target}",
            save_path=fig_lc
        )
    except Exception as e:
        fig_lc = ""
        print(f"[WARN] Learning curve failed for Stacked Ensemble → {target}: {e}")

    # Save CV table
    cv_df = pd.DataFrame(stack_gs.cv_results_)
    cv_path = os.path.join(TGT_TABLES, f"CV_StackedEnsemble_{sanitize(target)}.csv")
    cv_df.to_csv(cv_path, index=False)

    # Save model bundle
    mdl_path = os.path.join(TGT_MODELS, f"model_StackedEnsemble_{sanitize(target)}.pkl")
    bundle = {
        "pipeline": stack_best,
        "metadata": {
            "model_name": "Stacked Ensemble (RF+GB+GPR+DNN)",
            "target": target,
            "feature_names": feature_names,
            "family_order": FAMILY_ORDER,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "best_params": stack_best_params,
            "best_cv_rmse": stack_best_cv_rmse,
            "data_file": DATA_XLSX,
            "base_models": STACK_BASE_NAMES
        }
    }
    dump(bundle, mdl_path)

    summary_rows.append({
        "Model": "Stacked Ensemble",
        "Target": target,
        "Best CV RMSE": round(stack_best_cv_rmse, 6),
        "Train MAE": round(mae_tr, 6),
        "Train RMSE": round(rmse_tr, 6),
        "Train R²": round(r2_tr, 6),
        "Test MAE": round(mae_te, 6),
        "Test RMSE": round(rmse_te, 6),
        "Test R²": round(r2_te, 6),
        "Best Params": json.dumps(stack_best_params),
        "Model Path": mdl_path,
        "CV Table": cv_path,
        "Parity Plot": fig_parity,
        "Residual Plot": fig_resid,
        "Learning Curve": fig_lc
    })

# ----------------- SAVE SUMMARY -----------------
summary_df = pd.DataFrame(summary_rows)

tables_dir = os.path.join(RUN_DIR, "tables")
os.makedirs(tables_dir, exist_ok=True)

sum_csv  = os.path.join(tables_dir, "summary_models.csv")
sum_xlsx = os.path.join(tables_dir, "summary_models.xlsx")

summary_df.to_csv(sum_csv, index=False)
with pd.ExcelWriter(sum_xlsx, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="Summary")

print("\n=== DONE ===")
print("Run folder:", RUN_DIR)
print("Summary CSV:", sum_csv)
print("Summary XLSX:", sum_xlsx)
print(f"Total elapsed: {(time.time()-overall_t0)/60:.1f} min")

# Print summary
try:
    from IPython.display import display
    display(summary_df)
except Exception:
    print(summary_df.to_string(index=False))

print("\nTip: Plots are saved here (per target):")
for t in TARGETS:
    print(" -", os.path.join(RUN_DIR, sanitize(t), "plots"))