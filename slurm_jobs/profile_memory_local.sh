export OMP_NUM_THREADS=1
mpiexec -n 8 ~/venv-firedrake/bin/python ../main.py -log_view :memory_log.txt:ascii_info -log_view_memory