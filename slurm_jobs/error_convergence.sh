#!/bin/sh
#SBATCH --account maths
#SBATCH --time=12:00:00
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

# 4. The exact solution checkpoint is large, so keep it on /scratch rather than in
# the home directory. Point a later job at the same path with --exact to reuse it
# instead of paying for the fine solve again.
EXACT_FILE=/scratch/eltrob002/psi_exact.h5

# 5. Each (p, N) pair runs its own `mpiexec -n 40` (error_convergence.py forks one
# per data point so every solve gets the full node - see sweep.py for why).
apptainer exec \
    --bind /scratch/eltrob002 \
    --bind $HOST_CACHE_DIR \
    ~/firedrake.sif python3 Thesis/error_convergence.py \
    --job_id ${SLURM_JOB_ID} \
    --max_p 10 \
    --min_dofs 1000000 \
    --num_resolutions 20 \
    --max_dofs 10000000 \
    --exact_N 40 \
    --exact_p 7 \
    --ksp_rtol 1e-12 \
    --exact ${EXACT_FILE} \
    --ranks 40