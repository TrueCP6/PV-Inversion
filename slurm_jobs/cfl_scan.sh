#!/bin/sh
# One job per Courant number at n=40 p=4, each run to the same simulated time.
# The limiter is always on (prognostic_solver).
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

# cfl_scan_3 found nothing unstable up to 2.3. 1.3 (SAFETY) is the reference the others'
# roughness is judged against (timestepping.roughness_ratios), so it has to last the full run
COURANTS="1.3 1.7 1.9 2.1 2.3 2.5 2.7"
POINTS=$(echo $COURANTS | wc -w)
# The slowest run, C = 1.3, takes ~25 steps per simulated hour (scaled from cfl_scan_2's
# 2817 steps to 96 h) - ~35 if max(|u| + |v|) is sqrt(2) past max|u| - at 5-11 s a step:
# 2-6 minutes a simulated hour, so 168 h fits in ~18 h at worst
HOURS=168
N=40
P=4

if [ -z "$SLURM_JOB_ID" ]; then
    for TASK in $(seq 0 $((POINTS - 1))); do
        sbatch --export=ALL,TASK=$TASK "$0"
    done
    exit
fi

COURANT=$(echo $COURANTS | cut -d" " -f$((TASK + 1)))

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
