#!/bin/sh
#SBATCH --account maths
#SBATCH --time=72:00:00
#SBATCH --nodes=1 --ntasks=40
#SBATCH --mem=300G
#SBATCH --job-name="firedrake"
#SBATCH --mail-user=eltrob002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=firedrake_%j.out
#SBATCH --error=firedrake_%j.err
#SBATCH --constraint=large

# usage: sbatch timestepping.sh [PARAMS_JSON] - PhysicalParams overrides, e.g. param_sampler.py's
# optimised_*.json, as an absolute path; the default parameters if left out.
# At SAFETY = 1.3 the default run takes ~4400 steps of ~33 s to 168 h, hence the 72 h limit.
PARAMS=${1:+--params $1}

HOST_CACHE_DIR=/tmp/firedrake_cache_${SLURM_JOB_ID}
mkdir -p $HOST_CACHE_DIR

export APPTAINERENV_XDG_CACHE_HOME=$HOST_CACHE_DIR
export APPTAINERENV_PYOP2_CACHE_DIR=${HOST_CACHE_DIR}/pyop2

apptainer exec \
    --bind /scratch/eltrob002 \
    --bind $HOST_CACHE_DIR \
    ~/firedrake.sif \
    mpiexec -n ${SLURM_NTASKS} \
    python3 ~/Thesis/timestepping.py \
    --job_id ${SLURM_JOB_ID} \
    -p 4 \
    -T 168 \
    --backup 24 \
    $PARAMS
