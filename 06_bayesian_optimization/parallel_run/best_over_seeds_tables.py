import sys, glob, os
import pandas as pd

pd.set_option("display.width", 220)


def load(run):
    fs = glob.glob(os.path.join(run, "**", "bo_best_summary.csv"), recursive=True)
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "family_fixed" in df.columns:
        df["family"] = df["best_family"].where(df["best_family"].notna(), df["family_fixed"]) if "best_family" in df.columns else df["family_fixed"]
    return df


def tables(run, label):
    df = load(run)
    print(f"##### {label}: {len(df)} summaries; seeds {df['seed'].min()}-{df['seed'].max()}")
    des = ["UCx_used (mm)", "UCy_used (mm)", "UCz_used (mm)", "t_used (mm)"]
    for obj in ("YM", "SS"):
        m = df[(df["mode"] == "Mixed") & (df["objective"] == obj)]
        i = m["Best_Objective"].idxmax()
        print(f"-- GLOBAL (Mixed) best {obj} over 50 seeds:")
        print(m.loc[[i], ["seed", "best_family", "Best_Objective"] + des].round(5).to_string(index=False))
        print("   seeds reaching it (1e-6):", int((m["Best_Objective"] > m["Best_Objective"].max() - 1e-6).sum()), "of", len(m))
        p = df[(df["mode"] == "PerFamily") & (df["objective"] == obj)]
        idx = p.groupby("family_fixed")["Best_Objective"].idxmax()
        print(f"-- PER-FAMILY best {obj} over 50 seeds:")
        print(p.loc[idx, ["family_fixed", "seed", "Best_Objective"] + des].round(5).to_string(index=False))


if __name__ == "__main__":
    tables(sys.argv[1], sys.argv[2])
