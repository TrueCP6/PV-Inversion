"""Record the prognostic (DG + SSPRK3) solver evolving the control state, for
showcase/render_prognostic.py.

Uses timestepping.py's domain - the anomaly starts 2500 km from the inflow wall, and Lx is
stretched to hold everything the jet carries it through - at a resolution a 4-core machine
can step through in about an hour. Every --frame-hours of simulated time the state is
sampled onto regular grids and written to OUT/prognostic/frame_####.npz.

Run inside the Firedrake container (see showcase/README.md):
    mpiexec -n 4 python3 showcase/capture_prognostic.py --out /data
"""
from hash_seed import use_same_hash
use_same_hash()
import argparse
import os
import sys
import time
import numpy as np
from firedrake import Function
from firedrake.petsc import PETSc
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep
import timestepping
from derived_quantities import ResolvedAtmosphere
from parameters import PhysicalParams
from prognostic_solver import SAFETY
from sampling import GridSampler, axis
from tropopause import PVU

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--hours", type=float, default=168)
parser.add_argument("-p", type=int, default=2)
parser.add_argument("--ny", type=int, default=24)
parser.add_argument("--nz", type=int, default=24)
parser.add_argument("--courant", type=float, default=SAFETY)
parser.add_argument("--frame-hours", type=float, default=1.0)
parser.add_argument("--max-steps", type=int, default=None, help="Stop early, for timing")
parser.add_argument("--out", default="showcase_data")
args = parser.parse_args()

sweep.quiet_petsc()
rank0 = MPI.COMM_WORLD.rank == 0
T = args.hours * 3600

# timestepping._domain, with the resolution as a parameter
dx = timestepping.LY / args.ny
nx = int(np.ceil((2 * timestepping.ANOMALY_X + PhysicalParams.jet_magnitude * T) / dx))
Lx = nx * dx
phys = timestepping._phys_params(Lx)
solver = timestepping._build_solver(nx, args.ny, args.nz, args.p, phys)
atmos = solver.atmos
PETSc.Sys.Print(f"{nx}x{args.ny}x{args.nz} cells, p={args.p}, Lx={Lx / 1e3:.0f} km, "
                f"{solver.q.function_space().dim():,} DG DoF, "
                f"{atmos.cg_space.dim():,} CG DoF")

out_dir = os.path.join(args.out, "prognostic")
mesh = atmos.mesh
Ly, H = phys.Ly, phys.H
volume = GridSampler(mesh, axis(0, Lx, int(Lx / 100e3) + 1), axis(0, Ly, 51), axis(0, 20e3, 41))
surf_x, surf_y = axis(0, Lx, int(Lx / 50e3) + 1), axis(0, Ly, 101)
surface = GridSampler(mesh._base_mesh, surf_x, surf_y)
epv = Function(atmos.dg_space)

if rank0:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "meta.npz"),
                        x3=volume.axes[0], y3=volume.axes[1], z3=volume.axes[2],
                        surf_x=surf_x, surf_y=surf_y, Lx=Lx, Ly=Ly, H=H,
                        nx=nx, ny=args.ny, nz=args.nz, p=args.p,
                        dg_dofs=solver.q.function_space().dim(), cg_dofs=atmos.cg_space.dim(),
                        anomaly_x=phys.anomaly_x_pos, anomaly_y=phys.anomaly_y_pos)

def capture(frame, dt, wall, steps=0):
    solver.resolve()  # psi to match q, as timestepping._diagnostics does
    derived = ResolvedAtmosphere(solver.psi, atmos, solver.q)
    V2 = derived._surface_func_space()
    psi_s = derived._psi_surf()
    record = dict(
        t=solver.t, dt=dt, wall=wall, steps=steps,
        epv=volume(epv.interpolate(atmos.ertel_from_qgpv(solver.q) / PVU)),
        u_surf=surface(Function(V2).interpolate(-psi_s.dx(1))),
        v_surf=surface(Function(V2).interpolate(psi_s.dx(0))),
        p_surf=surface(derived.surf_pressure_ano_hpa()),
        min_p=derived.min_surf_pressure_ano_hpa(),
        max_wind=derived.max_surf_wind_speed(),
        z_dyn=derived.min_dyn_tropopause_height(),
        limited=solver.limiter.limited_fraction,
    )
    if rank0:
        np.savez_compressed(os.path.join(out_dir, f"frame_{frame:04d}.npz"), **record)
    PETSc.Sys.Print(f"frame {frame:4d}  t = {solver.t / 3600:6.2f} h  min p* = {record['min_p']:6.2f} hPa"
                    f"  max |u| = {record['max_wind']:5.2f} m/s  z_dyn = {record['z_dyn']:6.0f} m")

frame_seconds = args.frame_hours * 3600
started = time.perf_counter()
solver.resolve()
capture(0, float("nan"), 0.0)
frame, steps = 1, 0
while solver.t < T - 1e-6:
    t_next = frame * frame_seconds
    step_start = time.perf_counter()
    dt = min(solver.dt(safety=args.courant), t_next - solver.t)  # land exactly on each frame
    solver.step(dt)
    steps += 1
    PETSc.Sys.Print(f"step {steps:5d}  t = {solver.t / 3600:7.3f} h  dt = {dt:6.1f} s  "
                    f"({time.perf_counter() - step_start:.2f} s wall)")
    if solver.t >= t_next - 1e-6:
        capture(frame, dt, time.perf_counter() - started, steps)
        frame += 1
    if args.max_steps is not None and steps >= args.max_steps:
        break

PETSc.Sys.Print(f"done: {steps} steps, {frame} frames in {time.perf_counter() - started:.0f} s")
