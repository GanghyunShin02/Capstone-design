from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import os
import csv


# ============================================================
# 1. Mesh
# ============================================================

mesh = Mesh("/home/ss/Capstone/firedrake/meshfile/Navie.msh")

PETSc.Sys.Print("Number of cells:", mesh.num_cells())


# Gmsh Physical Curve tags
INLET = 1
TOP = 2
OUTLET = 3
BOTTOM = 4
CYLINDER = 5


# ============================================================
# 2. Function spaces
#    Taylor-Hood P2/P1
# ============================================================

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q


# ============================================================
# 3. Unknowns
# ============================================================

up = Function(W)
u, p = split(up)
v, q = TestFunctions(W)

u_old = Function(V, name="Velocity_old")
u_old.assign(Constant((0.0, 0.0)))


# ============================================================
# 4. Physical parameters
# ============================================================

nu = Constant(0.001)

H = 0.41
D = 0.1

U_mean_ref = 1.0


# ============================================================
# 5. Time
# ============================================================

t = Constant(0.0)

dt_value = 1.0 / 1600.0
dt = Constant(dt_value)

T = 8.0

time_value = 0.0
step = 0


# ============================================================
# 6. Inlet velocity
# ============================================================

x, y = SpatialCoordinate(mesh)

Umax = 1.5 * sin(pi * t / 8.0)

u_in = as_vector((
    4.0 * Umax * y * (H - y) / H**2,
    0.0
))


# ============================================================
# 7. Control variable: cylinder rotation
#
# omega = 0  -> original DFG no-slip cylinder
# omega > 0  -> counter-clockwise rotation
# omega < 0  -> clockwise rotation
# ============================================================

omega = Constant(0.0)

xc = Constant(0.2)
yc = Constant(0.2)

u_cylinder = as_vector((
    -omega * (y - yc),
     omega * (x - xc)
))


# ============================================================
# 8. Boundary conditions
# ============================================================

bc_inlet = DirichletBC(
    W.sub(0),
    u_in,
    INLET
)

bc_wall = DirichletBC(
    W.sub(0),
    Constant((0.0, 0.0)),
    (TOP, BOTTOM)
)

bc_cylinder = DirichletBC(
    W.sub(0),
    u_cylinder,
    CYLINDER
)

bcs = [
    bc_inlet,
    bc_wall,
    bc_cylinder
]

# OUTLET: natural do-nothing condition


# ============================================================
# 9. Crank-Nicolson
# ============================================================

u_mid = 0.5 * (u + u_old)


# ============================================================
# 10. Weak form
# ============================================================

F = (
    inner((u - u_old) / dt, v) * dx
    + nu * inner(grad(u_mid), grad(v)) * dx
    + inner(dot(grad(u_mid), u_mid), v) * dx
    - p * div(v) * dx
    + q * div(u) * dx
)


# ============================================================
# 11. Nonlinear problem / solver
# ============================================================

problem = NonlinearVariationalProblem(
    F,
    up,
    bcs=bcs
)

solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_converged_reason": None,
}

solver = NonlinearVariationalSolver(
    problem,
    solver_parameters=solver_parameters
)


# ============================================================
# 12. Solution views
# ============================================================

u_out, p_out = up.subfunctions

u_out.rename("Velocity")
p_out.rename("Pressure")


# ============================================================
# 13. Cylinder force
# ============================================================

n = FacetNormal(mesh)
I = Identity(2)

sigma = nu * grad(u_out) - p_out * I

# n points outward from the fluid domain.
# On the cylinder this points into the solid,
# so the minus sign gives force acting on the cylinder.
traction = -dot(sigma, n)


def calculate_coefficients():
    FD = assemble(
        traction[0] * ds(CYLINDER)
    )

    FL = assemble(
        traction[1] * ds(CYLINDER)
    )

    CD = (
        2.0 * FD
        / (U_mean_ref**2 * D)
    )

    CL = (
        2.0 * FL
        / (U_mean_ref**2 * D)
    )

    return float(CD), float(CL)


# ============================================================
# 14. Pressure probes
# ============================================================

probe_points = np.array([
    [0.15, 0.20],
    [0.25, 0.20]
])

probe = PointEvaluator(
    mesh,
    probe_points
)


def pressure_difference():
    values = probe.evaluate(p_out)

    p_front = float(values[0])
    p_back = float(values[1])

    return p_front - p_back


# ============================================================
# 15. Control / one-step FEM API
# ============================================================

def set_control(control=0.0):
    """
    Set cylinder angular velocity omega.

    control may be:
      - scalar: 0.25
      - list/tuple/ndarray: [0.25]

    This function only updates the Firedrake Constant.
    It does not advance the PDE.
    """

    if isinstance(control, (list, tuple, np.ndarray)):
        control = control[0]

    omega.assign(float(control))


def fem_step(control=0.0):
    """
    Advance the existing Firedrake DFG solver by exactly one timestep.

    Parameters
    ----------
    control:
        Cylinder angular velocity omega.
        control=0 reproduces the original stationary-cylinder benchmark BC.

    Returns
    -------
    dict with time, Cd, Cl, dp, omega
    """

    global time_value, step

    set_control(control)

    time_value += dt_value
    step += 1

    # Update time-dependent inlet
    t.assign(time_value)

    # Existing Firedrake FEM solve
    solver.solve()

    CD, CL = calculate_coefficients()
    dp = pressure_difference()

    # u^n <- u^(n+1)
    u_old.assign(u_out)

    return {
        "time": time_value,
        "Cd": CD,
        "Cl": CL,
        "dp": dp,
        "omega": float(omega),
    }


def get_observation():
    """
    Current low-dimensional observation.
    Later HydroGym can replace/extend this with velocity probes.
    """
    CD, CL = calculate_coefficients()
    dp = pressure_difference()

    return np.array([CD, CL, dp], dtype=float)


def reset_state():
    """
    Reset the CFD state to t=0 and zero velocity/pressure.
    Useful when an RL episode is restarted.
    """

    global time_value, step

    time_value = 0.0
    step = 0

    t.assign(0.0)
    omega.assign(0.0)

    up.assign(0.0)
    u_old.assign(Constant((0.0, 0.0)))


# ============================================================
# 16. Standalone benchmark runner
# ============================================================

def run_benchmark(
    final_time=T,
    output_dir="dfg2d3_output",
    save_interval=0.05,
    control=0.0,
):
    """
    Run the original benchmark as a standalone simulation.

    This is called only when this file is executed directly.
    Importing this module will NOT start the time loop.
    """

    reset_state()

    if mesh.comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    mesh.comm.barrier()

    vtk = VTKFile(
        os.path.join(output_dir, "dfg2d3.pvd")
    )

    vtk.write(
        u_out,
        p_out,
        time=0.0
    )

    csv_file = None
    writer = None

    if mesh.comm.rank == 0:
        csv_file = open(
            os.path.join(output_dir, "benchmark.csv"),
            "w",
            newline=""
        )

        writer = csv.writer(csv_file)

        writer.writerow([
            "time",
            "Cd",
            "Cl",
            "pressure_difference",
            "omega",
        ])

    save_every = max(
        1,
        int(round(save_interval / dt_value))
    )

    CD_max = -1.0e100
    CL_max = -1.0e100

    t_CD_max = 0.0
    t_CL_max = 0.0

    result = None

    try:
        while time_value < final_time - 0.5 * dt_value:
            result = fem_step(control)

            CD = result["Cd"]
            CL = result["Cl"]
            dp = result["dp"]

            if CD > CD_max:
                CD_max = CD
                t_CD_max = time_value

            if CL > CL_max:
                CL_max = CL
                t_CL_max = time_value

            if mesh.comm.rank == 0:
                writer.writerow([
                    time_value,
                    CD,
                    CL,
                    dp,
                    float(omega),
                ])

            if step % save_every == 0:
                vtk.write(
                    u_out,
                    p_out,
                    time=time_value
                )

                PETSc.Sys.Print(
                    f"t = {time_value:.4f}, "
                    f"Cd = {CD:.6f}, "
                    f"Cl = {CL:.6f}, "
                    f"dp = {dp:.6f}, "
                    f"omega = {float(omega):.6f}"
                )

        PETSc.Sys.Print("")
        PETSc.Sys.Print("========== DFG 2D-3 ==========")

        PETSc.Sys.Print(
            f"max Cd = {CD_max:.10f} "
            f"at t = {t_CD_max:.10f}"
        )

        PETSc.Sys.Print(
            f"max Cl = {CL_max:.10f} "
            f"at t = {t_CL_max:.10f}"
        )

        if result is not None:
            PETSc.Sys.Print(
                f"dp(t={time_value:.6f}) = {result['dp']:.10f}"
            )

    finally:
        if csv_file is not None:
            csv_file.close()


# ============================================================
# 17. Entry point
# ============================================================

def main():
    run_benchmark()


if __name__ == "__main__":
    main()
