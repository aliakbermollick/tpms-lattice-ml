#!/bin/bash
#SBATCH --job-name=tpms_bo
# #SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Parallel BO campaigns (50 seeds each) on one 40-core node.
# WORKDIR holds: code/ (the three BO scripts), models/ (model folders), venv/ (Python environment).
WORKDIR=${WORKDIR:-$HOME/tpms_bo}

# One BLAS thread per worker so the 40 joblib workers do not oversubscribe the node
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MODEL_RUN=$WORKDIR/models
export N_JOBS=${SLURM_CPUS_PER_TASK:-40}
PY=$WORKDIR/venv/bin/python
CODE=$WORKDIR/code

echo "start $(date)  node $(hostname)  cpus $N_JOBS"

mkdir -p $WORKDIR/out/gpr_50seeds $WORKDIR/out/se_50seeds
OUT_ROOT=$WORKDIR/out/gpr_50seeds $PY $CODE/bo_gpr_50seeds.py
echo "GPR done $(date)"
OUT_ROOT=$WORKDIR/out/se_50seeds $PY $CODE/bo_se_50seeds.py
echo "SE done $(date)"
