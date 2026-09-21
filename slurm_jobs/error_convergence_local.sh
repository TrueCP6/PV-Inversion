export OMP_NUM_THREADS=1
# 2 ranks puts the min_dofs floor at 2e5, low enough that the sweep still fits on a laptop.
mpiexec -n 2 ~/venv-firedrake/bin/python ../error_convergence.py \
  --job_id 1 \
  --max_p 6 \
  --max_dofs 350000 \
  --num_resolutions 3
