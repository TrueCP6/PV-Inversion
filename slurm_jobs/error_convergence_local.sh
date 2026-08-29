export OMP_NUM_THREADS=1
~/venv-firedrake/bin/python ../error_convergence.py \
  --job_id 1 \
  --ranks 8 \
  --max_p 6 \
  --max_dofs 350000 \
  --num_resolutions 3 \
  --exact_N 35 \
  --exact_p 4 \
  --exact psi_exact.h5