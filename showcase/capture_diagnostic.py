"""Record the diagnostic (QGPV inversion) solver at work, for showcase/render_diagnostic.py.

Inverts the control state exactly as the thesis does (matrix-free CG with p-multigrid), with a
KSP monitor that keeps every conjugate gradient iterate and the solver time it was reached at.
For contrast, the same system is then solved with textbook Jacobi-preconditioned CG on the
same cores. Iterates, inputs and converged fields are sampled onto regular grids and written
to OUT/diagnostic.npz.

Run inside the Firedrake container (see showcase/README.md):
    mpiexec -n 4 python3 showcase/capture_diagnostic.py --out /data
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
import gc
import os
import sys
import time
from dataclasses import replace
import numpy as np
from firedrake import (Function, LinearVariationalProblem, LinearVariationalSolver,
                       VectorSpaceBasis)
from firedrake.petsc import PETSc
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep
from barnes_atmosphere import BarnesAtmosphere
from derived_quantities import ResolvedAtmosphere
from math_utils import relative_error
from sampling import GridSampler, axis
from tropopause import PVU

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=40)
parser.add_argument("-p", type=int, default=4)
parser.add_argument("--jacobi-its", type=int, default=3000,
                    help="Iteration cap for the Jacobi-preconditioned comparison (0 to skip it)")
parser.add_argument("--out", default="showcase_data")
args = parser.parse_args()

sweep.quiet_petsc()
rank0 = MPI.COMM_WORLD.rank == 0
data = {}

start = time.perf_counter()
solver = sweep.build_solver(args.N, args.p, matfree=True)
atmos = solver.atmos
V = solver.cg_space
pp = atmos.phys_params
data["setup_seconds"] = time.perf_counter() - start
data["dofs"] = V.dim()
data["N"], data["p"] = args.N, args.p
data["ranks"] = MPI.COMM_WORLD.size
PETSc.Sys.Print(f"{V.dim():,} DoF")

# First solve compiles every kernel; the timed solve is a repeat from zero, like every solve
# after the first in a parameter sweep
solver.solve_psi()
data["solve_seconds"] = solver.solve_psi(zero_initial_guess=True)


def collect():
    """Firedrake objects sit in reference cycles and PETSc destroys are deferred, so free them
    explicitly - otherwise thousands of monitored iterations exhaust memory."""
    gc.collect()
    PETSc.garbage_cleanup(PETSc.COMM_WORLD)


class Monitor:
    """A KSP monitor that records the residual and the solver time at every iteration, and
    hands the iterates it is asked to keep to `visit` - with the time spent there left off
    the clock.

    Firedrake's linear solve is a KSPONLY SNES: it solves J y = F(psi_0) and sets
    psi = psi_0 - y, so from a zero start the iterate is -y.
    """
    def __init__(self, ksp, visit, keep=lambda its: True):
        self.visit, self.keep = visit, keep
        self.residuals, self.times = [], []
        self.solver_time = 0.0
        self._resumed = None
        ksp.setMonitor(self)

    def __call__(self, ksp, its, rnorm):
        now = time.perf_counter()
        if self._resumed is not None:
            self.solver_time += now - self._resumed
        self.residuals.append(rnorm)
        self.times.append(self.solver_time)
        if self.keep(its):
            psi = Function(V)
            with psi.dat.vec_wo as v:
                ksp.buildSolution().copy(v)
            psi.assign(-psi)
            self.visit(its, psi)
        self._resumed = time.perf_counter()


# The monitored p-multigrid solve, as configured in parameters.SolverParams.matfree_params
iterates = []
ksp = solver.solver.snes.ksp
pmg = Monitor(ksp, lambda its, psi: iterates.append(psi))
solver.psi_soln.assign(0)
solver.solver.solve()
ksp.monitorCancel()
final = solver.psi_soln.copy(deepcopy=True)
PETSc.Sys.Print(f"p-multigrid: {len(iterates) - 1} iterations, {pmg.solver_time:.2f} s")
data["pmg_residuals"] = np.array(pmg.residuals)
data["pmg_times"] = np.array(pmg.times)
data["pmg_errors"] = np.array([relative_error(final, psi) if k else 1.0
                               for k, psi in enumerate(iterates)])

# Grids. The thesis plots crop to the central 5000 km, where everything happens.
Lx, Ly, H = pp.Lx, pp.Ly, pp.H
xc, yc = pp.anomaly_x_pos, pp.anomaly_y_pos
mesh = atmos.mesh
crop = 2500e3
x3, y3, z3 = axis(xc - crop, xc + crop, 81), axis(yc - crop, yc + crop, 81), axis(0, H, 61)
volume = GridSampler(mesh, x3, y3, z3)
xs, zs = axis(xc - crop, xc + crop, 241), axis(0, H, 121)
slice_xz = GridSampler(mesh, xs, np.array([yc]), zs)
slice_yz = GridSampler(mesh, np.array([xc]), xs - xc + yc, zs)
surf_x, surf_y = axis(0, Lx, 201), axis(0, Ly, 201)
surface = GridSampler(mesh._base_mesh, surf_x, surf_y)
data.update(x3=x3, y3=y3, z3=z3, xs=xs, zs=zs, surf_x=surf_x, surf_y=surf_y)

# Inputs: what the solver is given
cg = lambda expr: Function(V).interpolate(expr)
epv = cg(atmos.ertel_pv() / PVU)
data["epv_volume"] = volume(epv)
data["epv_xz"] = slice_xz(epv)
data["epv_yz"] = slice_yz(epv)
jet = cg(atmos.u())
data["jet_volume"] = volume(jet)
data["jet_yz"] = slice_yz(jet)


def sections(psi):
    """Pressure anomaly on the two vertical sections and the surface, plus surface wind."""
    derived = ResolvedAtmosphere(psi, atmos)
    V2 = derived._surface_func_space()
    psi_s = derived._psi_surf()
    p_star = derived.pressure_anomaly_hpa()
    return dict(p_xz=slice_xz(p_star), p_yz=slice_yz(p_star),
                p_surf=surface(derived.surf_pressure_ano_hpa()),
                u_surf=surface(Function(V2).interpolate(-psi_s.dx(1))),
                v_surf=surface(Function(V2).interpolate(psi_s.dx(0))))


def error_sections(psi):
    """Pressure anomaly of psi - final. p* is linear in psi, so this is the error in p*,
    sampled from the difference rather than differenced after rounding to float32."""
    err = sections(Function(V).assign(psi - final))
    return {f"err_{key}": value for key, value in err.items() if key.startswith("p_")}


def stack(snaps, prefix):
    if rank0:
        for key in snaps[0]:
            data[f"{prefix}_{key}"] = np.stack([s[key] for s in snaps])


snaps = []
while iterates:
    psi = iterates.pop(0)
    snaps.append({**sections(psi), **error_sections(psi)})
    del psi
    collect()
stack(snaps, "pmg")
PETSc.Sys.Print("sampled p-multigrid iterates")

# Converged state, with the extra fields the 3D renders use
for key, value in sections(final).items():
    data[f"final_{key}"] = value
derived = ResolvedAtmosphere(final, atmos)
data["final_p_volume"] = volume(derived.pressure_anomaly_hpa())
data["final_speed_volume"] = volume(derived.horizontal_wind_speed())
data["final_speed_yz"] = slice_yz(derived.horizontal_wind_speed())
data["final_theta_xz"] = slice_xz(derived.potential_temperature())
data["final_theta_yz"] = slice_yz(derived.potential_temperature())
data["min_surf_p"] = derived.min_surf_pressure_ano_hpa()
data["max_surf_wind"] = derived.max_surf_wind_speed()
data["z_dyn"] = derived.min_dyn_tropopause_height()
data["rossby"] = derived.rossby_number()

# The same system with textbook Jacobi-preconditioned CG, on the same cores. Iterates are
# sampled on a roughly geometric schedule, since there are thousands of them.
if args.jacobi_its:
    keep = set(range(11)) | set(np.unique(np.geomspace(10, args.jacobi_its, 70).astype(int)).tolist())
    a, L = solver._specify_equation()
    psi_jacobi = Function(V)
    jacobi = LinearVariationalSolver(
        LinearVariationalProblem(a, L, psi_jacobi, constant_jacobian=True,
                                 form_compiler_parameters=solver.solver_params.form_compiler_params),
        solver_parameters={"mat_type": "matfree", "ksp_type": "cg", "pc_type": "jacobi",
                           "ksp_rtol": solver.solver_params.ksp_rtol,
                           "ksp_atol": solver.solver_params.ksp_atol,
                           "ksp_max_it": args.jacobi_its},
        nullspace=VectorSpaceBasis(constant=True, comm=mesh.comm))
    jacobi_ksp = jacobi.snes.ksp
    # One iteration to compile and assemble the diagonal, off the clock
    jacobi_ksp.setTolerances(max_it=1)
    try:
        jacobi.solve()
    except Exception:
        pass
    jacobi_ksp.setTolerances(max_it=args.jacobi_its)
    psi_jacobi.assign(0)

    # Only what the race needs: the north-south section, and the error
    kept, kept_its, kept_errors = [], [], []
    def visit(its, psi):
        kept_its.append(its)
        kept_errors.append(relative_error(final, psi) if its else 1.0)
        kept.append({"p_yz": slice_yz(ResolvedAtmosphere(psi, atmos).pressure_anomaly_hpa())})
        PETSc.Sys.Print(f"Jacobi iteration {its}: error {kept_errors[-1]:.3e}")
        collect()

    monitor = Monitor(jacobi_ksp, visit, keep=lambda its: its in keep)
    try:
        jacobi.solve()
    except Exception as e:  # ConvergenceError once it hits the cap
        PETSc.Sys.Print(f"Jacobi stopped: {type(e).__name__}")
    jacobi_ksp.monitorCancel()
    # The cap is hit inside KSPSolve, so the last iterate may not have been kept
    data["jacobi_residuals"] = np.array(monitor.residuals)
    data["jacobi_times"] = np.array(monitor.times)
    data["jacobi_kept_its"] = np.array(kept_its)
    data["jacobi_errors"] = np.array(kept_errors)
    stack(kept, "jacobi")
    PETSc.Sys.Print(f"Jacobi: {len(monitor.residuals) - 1} iterations, {monitor.solver_time:.1f} s")

# The response to the intrusion alone. The jet carries a pressure dipole of its own that would
# hide the intrusion's low in a 3D view; the inversion is linear, so inverting the background
# by itself and subtracting leaves exactly the part the anomaly induces.
solver.update_atmosphere(BarnesAtmosphere(atmos.domain, replace(pp, anomaly_mag=0.0)))
solver.solve_psi()
induced = ResolvedAtmosphere(Function(V).assign(final - solver.psi_soln), atmos)
data["induced_p_volume"] = volume(induced.pressure_anomaly_hpa())
data["induced_speed_volume"] = volume(induced.horizontal_wind_speed())
data["induced_p_xz"] = slice_xz(induced.pressure_anomaly_hpa())

if rank0:
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "diagnostic.npz")
    np.savez_compressed(path, **data)
    print(f"wrote {path}: solve {data['solve_seconds']:.2f} s, "
          f"{len(data['pmg_residuals']) - 1} iterations, min p* {data['min_surf_p']:.2f} hPa, "
          f"max wind {data['max_surf_wind']:.2f} m/s")
