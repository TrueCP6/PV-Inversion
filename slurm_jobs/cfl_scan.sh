#!/bin/sh
# One array task per (configuration, Courant number): 2 configurations x 13 Courant numbers.
#SBATCH --account maths
#SBATCH --array=0-25
#SBATCH --time=16:00:00
#SBATCH --nodes=1 --ntasks=40
#SBATCH --mem=300G
#SBATCH --job-name="firedrake"
#SBATCH --mail-user=eltrob002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=firedrake_%A_%a.out
#SBATCH --error=firedrake_%A_%a.err
#SBATCH --constraint=large

POINTS=13 # Courant 0.3, 0.4, ..., 1.5
CONFIG=$((SLURM_ARRAY_TASK_ID / POINTS))
COURANT=$(awk "BEGIN { print 0.3 + 0.1 * ($SLURM_ARRAY_TASK_ID % $POINTS) }")

if [ "$CONFIG" -eq 0 ]; then N=80; P=2; else N=40; P=4; fi

HOST_CACHE_DIR=/tmp/firedrake_cache_${SLURM_JOB_ID}
mkdir -p $HOST_CACHE_DIR

export APPTAINERENV_XDG_CACHE_HOME=$HOST_CACHE_DIR
export APPTAINERENV_PYOP2_CACHE_DIR=${HOST_CACHE_DIR}/pyop2

apptainer exec \
    --bind /scratch/eltrob002 \
    --bind $HOST_CACHE_DIR \
    ~/firedrake.sif \
    mpiexec -n 40 \
    python3 ~/Thesis/timestepping.py \
    --job_id ${SLURM_JOB_ID} \
    --cfl-scan \
    --scan-n $N \
    -p $P \
    --scan-range $COURANT $COURANT \
    --scan-points 1 \
    --scan-steps 2000

# The JSON records neither n nor p, so put them in the name
mv cfl_scan_${SLURM_JOB_ID}.json cfl_scan_n${N}_p${P}_c${COURANT}.json
