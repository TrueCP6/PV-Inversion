from config import DomainConfig, PhysicsConfig, AnomalyConfig, SolverConfig
from atmosphere import AtmosphereBackground
from model import PVInversionModel

from firedrake import *


def create_domain(domain: DomainConfig):
    base_mesh = RectangleMesh(domain.N_h, domain.N_h, domain.x_max, domain.y_max)
    mesh = ExtrudedMesh(base_mesh, layers=domain.N_z, layer_height=(domain.z_top / domain.N_z))
    V = FunctionSpace(mesh, "CG", 1)

    PETSc.Sys.Print("Created extruded mesh")

    return mesh, V

def main(): #todo mpiexec script
    PETSc.Sys.Print("Starting setup...")

    # 1. Initialize configurations
    domain_cfg = DomainConfig(N_h=30, N_z=100)
    phys_cfg = PhysicsConfig()
    anom_cfg = AnomalyConfig()
    sol_cfg = SolverConfig()

    # 2. Build the mesh
    mesh, function_space = create_domain(domain_cfg)

    # 3. Setup the background physics
    atm = AtmosphereBackground(mesh, function_space, phys_cfg, domain_cfg)

    # 4. Initialize and run the model
    model = PVInversionModel(mesh, function_space, atm, domain_cfg, anom_cfg, sol_cfg)

    model.solve()
    model.save_output()

    PETSc.Sys.Print("Simulation complete.")


    #todo move into plotting file
    import pyvista as pv

    # 1. Load and slice the mesh
    mesh = pv.read("output/output_0.vtu") #todo open .pvd file
    slice_yz = mesh.slice(normal='x', origin=(500e3, 500e3, 10000))
    slice_yz["PV (PVU)"] = slice_yz["Potential Vorticity"] * 1e6

    # 2. Create a Plotter object for advanced rendering controls
    plotter = pv.Plotter()

    # 3. Add the slice to the plotter
    plotter.add_mesh(
        slice_yz,
        scalars="PV (PVU)",
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={'title': 'PV (PVU)'}
    )

    # 4. Snap camera to the Y-Z plane
    plotter.camera_position = 'yz'

    # 5. Scale the Z-axis visually
    # 1,000,000 m width / 20,000 m height = 50
    plotter.set_scale(zscale=25)

    # 6. Render the plot
    plotter.show()

if __name__ == "__main__":
    main()