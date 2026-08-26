#!/bin/sh
#SBATCH --account maths
#SBATCH --time=3:00:00
#SBATCH --nodes=1 --ntasks=40
#SBATCH --mem=300G
#SBATCH --job-name="firedrake"
#SBATCH --mail-user=eltrob002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=firedrake_%j.out
#SBATCH --error=firedrake_%j.err

# 1. Setup Apptainer environment
mkdir -p /scratch/eltrob002/apptainer_cache
mkdir -p /scratch/eltrob002/apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/eltrob002/apptainer_cache
export APPTAINER_TMPDIR=/scratch/eltrob002/apptainer_tmp

# 2. Create a unique, node-local cache directory on the host's /tmp/
HOST_CACHE_DIR=/tmp/firedrake_cache_${SLURM_JOB_ID}
mkdir -p $HOST_CACHE_DIR

# 3. Direct Apptainer to use this new local cache
export APPTAINERENV_XDG_CACHE_HOME=$HOST_CACHE_DIR
export APPTAINERENV_PYOP2_CACHE_DIR=${HOST_CACHE_DIR}/pyop2

# 5. Each resolution runs its own `mpiexec -n 40` (time_complexity.py forks one per
# data point so every test gets the full node - see time_complexity.py for why).
# time_complexity.py loops over resolutions itself, so tests run sequentially.
apptainer exec \
    --bind /scratch/eltrob002 \
    --bind $HOST_CACHE_DIR \
    ~/firedrake.sif python3 Thesis/time_complexity.py \
    --job_id ${SLURM_JOB_ID} \
    --num_solves 2 \
    --max_dofs_assembled 12000000 \
    --max_dofs_matfree 12000000 \
    --num_resolutions 20 \
    --ranks 40 \
    --num_initial_solves 3
