from firedrake import RectangleMesh, ExtrudedMesh, PETSc
from config import DomainConfig, PhysicsConfig, AnomalyConfig, SolverConfig
from atmosphere import AtmosphereBackground
from model import PVInversionModel


def create_mesh(domain: DomainConfig):
    base_mesh = RectangleMesh(domain.N_x, domain.N_x, domain.x_max, domain.y_max)
    mesh = ExtrudedMesh(base_mesh, layers=domain.N_x, layer_height=(domain.z_top / domain.N_x))
    return mesh


def main():
    PETSc.Sys.Print("Starting setup...")

    # 1. Initialize configurations
    domain_cfg = DomainConfig(N_x=30)
    phys_cfg = PhysicsConfig()
    anom_cfg = AnomalyConfig()
    sol_cfg = SolverConfig()

    # 2. Build the mesh
    mesh = create_mesh(domain_cfg)
    PETSc.Sys.Print("Created extruded mesh")

    # 3. Setup the background physics
    atm = AtmosphereBackground(mesh, phys_cfg, domain_cfg)

    # 4. Initialize and run the model
    model = PVInversionModel(mesh, atm, domain_cfg, anom_cfg, sol_cfg)

    model.solve()
    model.save_output()

    PETSc.Sys.Print("Simulation complete.")


if __name__ == "__main__":
    main()