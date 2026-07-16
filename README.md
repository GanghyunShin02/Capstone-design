# Capstone-design

This repo is for capstone design.


msh_tri = mesh.create_unit_square(
    MPI.COMM_WORLD, 8, 8,
    cell_type=mesh.CellType.triangle      # → 128 cells
)
msh_quad = mesh.create_unit_square(
    MPI.COMM_WORLD, 8, 8,
    cell_type=mesh.CellType.quadrilateral  # → 64 cells
)
# Rectangle [0,2]×[0,0.5] — 16×4 grid → 128 triangle cells
msh_rect = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0., 0.]), np.array([2., 0.5])], [16, 4]
)
# Unit cube [0,1]^3 — 4×4×4 → 384 tet cells
msh_3d = mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)


# Default diagonal direction
msh_right = mesh.create_unit_square(
    MPI.COMM_WORLD, 4, 4,
    diagonal=mesh.DiagonalType.right
)
# Left diagonal
msh_left = mesh.create_unit_square(
    MPI.COMM_WORLD, 4, 4,
    diagonal=mesh.DiagonalType.left
)
# Crossed (Union Jack pattern)
msh_crossed = mesh.create_unit_square(
    MPI.COMM_WORLD, 4, 4,
    diagonal=mesh.DiagonalType.crossed
