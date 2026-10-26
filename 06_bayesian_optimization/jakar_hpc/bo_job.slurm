#!/bin/bash
#SBATCH --job-name=tpms_bo
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=/work/aamollick/tpms_bo/out/slurm-%j.out

# One BLAS thread per worker so 40 joblib workers do not oversubscribe the node
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MODEL_RUN=/work/aamollick/tpms_bo/models
export N_JOBS=${SLURM_CPUS_PER_TASK:-40}
PY=/work/aamollick/tpms_bo/venv39b/bin/python
CODE=/work/aamollick/tpms_bo/code

echo "start $(date)  node $(hostname)  cpus $N_JOBS"

mkdir -p /work/aamollick/tpms_bo/out/gpr_50seeds /work/aamollick/tpms_bo/out/se_50seeds
OUT_ROOT=/work/aamollick/tpms_bo/out/gpr_50seeds $PY $CODE/bo_gpr_50seeds.py
echo "GPR done $(date)"
OUT_ROOT=/work/aamollick/tpms_bo/out/se_50seeds $PY $CODE/bo_se_50seeds.py
echo "SE done $(date)"
