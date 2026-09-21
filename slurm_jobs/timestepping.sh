#!/bin/sh
#SBATCH --account maths
#SBATCH --time=8:00:00
#SBATCH --nodes=1 --ntasks=40
#SBATCH --mem=300G
#SBATCH --job-name="firedrake"
#SBATCH --mail-user=eltrob002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=firedrake_%j.out
#SBATCH --error=firedrake_%j.err

HOST_CACHE_DIR=/tmp/firedrake_cache_${SLURM_JOB_ID}
mkdir -p $HOST_CACHE_DIR

export APPTAINERENV_XDG_CACHE_HOME=$HOST_CACHE_DIR
export APPTAINERENV_PYOP2_CACHE_DIR=${HOST_CACHE_DIR}/pyop2

apptainer exec \
    --bind $HOST_CACHE_DIR \
    ~/firedrake.sif \
    mpiexec -n 40 \
    python3 ~/Thesis/timestepping.py \
    --job_id ${SLURM_JOB_ID} \
    -n 80 \
    -p 2 \
    -T 168
