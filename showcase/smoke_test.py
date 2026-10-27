"""Quick check that the solver runs: build the control state, invert it, print the headline numbers."""
from hash_seed import use_same_hash
use_same_hash()
import sys
import time
from firedrake.petsc import PETSc
import sweep
from derived_quantities import ResolvedAtmosphere

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
p = int(sys.argv[2]) if len(sys.argv) > 2 else 4

sweep.quiet_petsc()
start = time.perf_counter()
solver = sweep.build_solver(N, p, matfree=True)
PETSc.Sys.Print(f"setup {time.perf_counter() - start:.2f} s, DoF {solver.psi_soln.function_space().dim():,}")
solver.solve_psi()
solver.solve_psi(zero_initial_guess=True)
ksp = solver.solver.snes.ksp
PETSc.Sys.Print(f"CG iterations: {ksp.getIterationNumber()}")
derived = ResolvedAtmosphere(solver.psi_soln, solver.atmos)
PETSc.Sys.Print(f"min surface p* = {derived.min_surf_pressure_ano_hpa():.3f} hPa, "
                f"max surface wind = {derived.max_surf_wind_speed():.3f} m/s, "
                f"z_dyn = {derived.min_dyn_tropopause_height():.0f} m, Ro = {derived.rossby_number():.3f}")
PETSc.Sys.Print(f"total {time.perf_counter() - start:.2f} s")
