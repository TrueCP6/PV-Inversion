from firedrake import TrialFunction, TestFunction, dx, DirichletBC, Constant, solve, Function, conditional

def compute_vertical_integral(integrand, V):
    """
    Computes the indefinite vertical integral of an arbitrary UFL integrand
    from z=0 to z. Solves the ODE: dI/dz = integrand with I(z=0) = 0.
    """
    I_trial = TrialFunction(V)
    W_test = TestFunction(V)

    a_I = I_trial.dx(2) * W_test * dx
    L_I = integrand * W_test * dx

    bc_I = DirichletBC(V, Constant(0.0), "bottom")
    I_func = Function(V)

    # todo check if this is the most optimal solver
    # GMRES is required for a first derivative (asymmetric matrix)
    int_solver_params = {
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu"
    }

    solve(a_I == L_I, I_func, bcs=[bc_I], solver_parameters=int_solver_params)
    return I_func

def kink_function(x, delta):
    """
    Returns a UFL expression for the kink profile kappa_delta(x).
    """
    # Define the bounds and values for each segment
    val_less_than_0 = Constant(0)
    val_lower_mid = 0.5 * (2 * x) ** delta
    val_upper_mid = 1 - 0.5 * (2 * (1 - x)) ** delta
    val_greater_than_1 = Constant(1)

    # Build the nested conditionals (evaluated from outside in)
    return conditional(x <= 0.0, val_less_than_0,
           conditional(x <= 0.5, val_lower_mid,
           conditional(x <= 1.0, val_upper_mid, val_greater_than_1)))

def scaled_kink(x, delta, left_val, right_val, kink_width, kink_centre):
    return (right_val - left_val) * kink_function((x-kink_centre)/kink_width + 0.5, delta) + left_val