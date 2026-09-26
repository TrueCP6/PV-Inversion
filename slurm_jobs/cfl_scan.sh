#!/bin/sh
# One job per (configuration, Courant number): 2 configurations x 5 Courant numbers,
# each run to the same simulated time.
# Run with `sh cfl_scan.sh` (not sbatch): it submits itself once per task,
# since array jobs never get scheduled on this cluster.
#SBATCH --account maths
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks=40
#SBATCH --mem=300G
#SBATCH --job-name="firedrake"
#SBATCH --mail-user=eltrob002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=firedrake_%j.out
#SBATCH --error=firedrake_%j.err

POINTS=5 # Courant 0.3, 0.4, ..., 0.7
# Roughly the time the old 2000-step scan covered at C = 0.3 (~2000 steps for n=40 p=4 at
# ~40 m/s peak wind), so the slowest run still fits in the wall time
HOURS=96

if [ -z "$SLURM_JOB_ID" ]; then
    for TASK in $(seq 0 $((2 * POINTS - 1))); do
        sbatch --export=ALL,TASK=$TASK "$0"
    done
    exit
fi

CONFIG=$((TASK / POINTS))
COURANT=$(awk "BEGIN { print 0.3 + 0.1 * ($TASK % $POINTS) }")

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
    --scan-hours $HOURS

# The JSON records neither n nor p, so put them in the name
mv cfl_scan_${SLURM_JOB_ID}.json cfl_scan_n${N}_p${P}_c${COURANT}.json
