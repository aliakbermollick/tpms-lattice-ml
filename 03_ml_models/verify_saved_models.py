"""Re-score saved models on the fixed 80/20 stratified split (random_state=42) and print Table 3 metrics.
Usage: python verify_saved_models.py <run_dir> [data.xlsx]   (needs scikit-learn==1.5.2, joblib, pandas, openpyxl)
"""
import sys, warnings, ast, numpy as np, pandas as pd, joblib, sklearn
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
base=sys.argv[1]; data=sys.argv[2] if len(sys.argv)>2 else __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"..","data","TPMS_data_27.12.25.xlsx")
print("sklearn",sklearn.__version__)
df=pd.read_excel(data)
NUM=["Thickness (t) (mm)","UCx (mm)","UCy (mm)","UCz (mm)"]
enc=np.vstack(df["Structure Encoding"].apply(ast.literal_eval).values).astype(float)
X=np.hstack([enc,df[NUM].values]); fam=df["Lattice Type"].astype(str).str.strip().values
Xtr,Xte,ftr,fte,itr,ite=train_test_split(X,fam,np.arange(len(df)),test_size=0.2,random_state=42,stratify=fam)
print("train/test",len(itr),len(ite))
models=["Decision_Tree","Random_Forest","Gradient_Boosting","GPR","ANN","DNN","StackedEnsemble"]
for tgt,folder,col in (("Youngs_Modulus_MPa","Youngs_Modulus_MPa","Young's Modulus (MPa)"),("Specific_Stiffness_MPa-kg-m3","Specific_Stiffness_MPa-kg-m3","Specific Stiffness (MPa/(kg/m3))")):
    y=df[col].values
    for m in models:
        try:
            b=joblib.load(f"{base}/{folder}/models/model_{m}_{tgt}.pkl")
            p=b["pipeline"].predict(Xte); yt=y[ite]
            print(f"{col[:6]} {m:18s} MAE {mean_absolute_error(yt,p):.4f} RMSE {np.sqrt(np.mean((yt-p)**2)):.4f} R2 {r2_score(yt,p):.4f}")
        except Exception as e: print(m,"FAIL",type(e).__name__,str(e)[:120])
