import json
from dataclasses import asdict
import numpy as np
from parameters import PhysicalParams, SolverParams

def quiet_petsc():
    """Stop PETSc reporting the options Firedrake sets on our behalf but never reads."""
    from firedrake.petsc import PETSc
    PETSc.Options().setValue("options_left", "false")

def is_main_rank():
    """True on the one rank that should write results out."""
    from mpi4py import MPI
    return MPI.COMM_WORLD.rank == 0

def dof_count(p : int, N : int):
    """Degrees of freedom of a Q_p space on an N x N x N mesh."""
    return (p * N + 1) ** 3

MIN_DOFS_PER_RANK = 100000

def min_dofs():
    """Smallest sweep point worth running on the ranks this job was launched with."""
    from mpi4py import MPI
    return MIN_DOFS_PER_RANK * MPI.COMM_WORLD.size

def resolutions_for_dofs(max_dofs : int, num_resolutions : int, p : int):
    """Resolutions whose dof counts are geometrically spaced from min_dofs() to max_dofs."""
    floor = min_dofs()
    if max_dofs < floor:
        raise ValueError(f"max_dofs of {max_dofs:.3g} is below the {floor:.3g} dof floor for "
                         f"this many ranks - ask for fewer ranks or a larger max_dofs")

    dofs = np.geomspace(floor, max_dofs, num_resolutions)
    return np.unique(np.round((np.cbrt(dofs) - 1) / p).astype(int))

def build_solver(N : int, p : int, matfree : bool, ksp_rtol : float = SolverParams.ksp_rtol,
                 quadrature_degree : int = SolverParams.quadrature_degree, atmos_cls = None):
    """Set up the psi solver on an N x N x N mesh of Q_p elements, over the Barnes atmosphere
    unless atmos_cls asks for another one (MMSChecker, for the error convergence sweep)."""
    import gc
    from firedrake.petsc import PETSc
    from domain_builder import DomainBuilder
    from barnes_atmosphere import BarnesAtmosphere
    from diagnostic_solver import DiagnosticSolver

    # Firedrake objects sit in reference cycles, so a finished solver lingers until the cyclic
    # collector happens to run - collect it before building the next, or both are held at once
    gc.collect()
    PETSc.garbage_cleanup(PETSc.COMM_WORLD)

    phys_params = PhysicalParams()
    solver_params = SolverParams(nx=N, ny=N, nz=N, check_flux=False, polynomial_order=p,
                                 ksp_rtol=ksp_rtol, quadrature_degree=quadrature_degree)

    domain = DomainBuilder(solver_params, phys_params)
    atmos = (atmos_cls or BarnesAtmosphere)(domain)

    return DiagnosticSolver(atmos, matfree)

def save_records(path, records, indent=None):
    """Write a list of dataclass records out as JSON, from the main rank only."""
    if not is_main_rank():
        return

    with open(path, "w") as f:
        json.dump([asdict(record) for record in records], f, indent=indent)

def load_records(path, record_cls):
    """Read a JSON results file back into a list of record_cls instances."""
    with open(path) as f:
        return [record_cls(**record) for record in json.load(f)]

def add_common_arguments(parser):
    """Options every sweep script takes."""
    parser.add_argument('-j', '--job_id', type=int, default=0)
    parser.add_argument('--plot', metavar='JSON_PATH', help='Plot the given results file instead of generating new data, then exit')
