export OMP_NUM_THREADS=1
mpiexec -n 8 ~/venv-firedrake/bin/python time_complexity.py \
  --job_id 1 \
  --num_solves 2 \
  --max_dofs_assembled 3000000 \
  --max_dofs_matfree 4000000 \
  --num_resolutions 5 \
  --num_initial_solves 2
