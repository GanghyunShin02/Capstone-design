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





from dolfinx.io import gmshio
import gmsh
gmsh.initialize()
# Create geometry
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)  # unit disk
gmsh.model.occ.synchronize()
# Set mesh size
gmsh.model.mesh.setSize(
    gmsh.model.getEntities(0), 0.1
)
# Generate 2D mesh
gmsh.model.mesh.generate(2)
# Import into DOLFINx
msh, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
)
gmsh.finalize()








from dolfinx import mesh
from mpi4py import MPI
import numpy as np
msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
tdim = msh.topology.dim
# Create all entities and connectivities
for d in range(tdim + 1):
    msh.topology.create_entities(d)
msh.topology.create_connectivity(tdim, 0)
msh.topology.create_connectivity(tdim, tdim - 1)
# Print summary
for d in range(tdim + 1):
    n = msh.topology.index_map(d).size_local
    names = {0: "vertices", 1: "edges", 2: "cells"}
    print(f"  dim {d} ({names[d]}): {n}")
# Print first 3 cells
c2v = msh.topology.connectivity(tdim, 0)
for c in range(min(3, msh.topology.index_map(tdim).size_local)):
    verts = [int(x) for x in c2v.links(c)]
    coords = msh.geometry.x[msh.geometry.dofmap[c], :2]
    print(f"  Cell {c}: verts={verts}, "
          f"coords={coords.tolist()}")
