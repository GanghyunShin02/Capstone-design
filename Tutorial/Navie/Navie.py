import numpy as np
import matplotlib.pyplot as plt

import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx import fem
from dolfinx import mesh,io
from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio
from dolfinx.mesh import (locate_entities_boundary,
                            create_submesh)
import ufl
from ufl import (grad,
                 dot,
                 inner,
                 TrialFunction,
                 TestFunction,
                 dx, lhs,
                 nabla_grad,
                 div, rhs,
                 sym)
from dolfinx.fem import (Function, 
                         functionspace,
                         dirichletbc,
                         locate_dofs_topological,
                         form,
                         Constant,
                         extract_function_spaces)
from dolfinx.fem.petsc import (assemble_matrix,
                               assemble_vector,
                               apply_lifting, 
                               create_matrix, 
                               create_vector,
                               set_bc)

from CoolProp.CoolProp import PropsSI
from pathlib import Path

from dolfinx.fem.petsc import create_vector as create_vector_petsc
import inspect
import time
start=time.time()


meshdata=gmshio.read_from_msh('/home/sgh/Navie.msh',MPI.COMM_WORLD,0,2)
domain=meshdata.mesh

cell_tags=meshdata.cell_tags
facet_tags=meshdata.facet_tags



inlettag=facet_tags.find(1)
uptag=facet_tags.find(2)
outlettag=facet_tags.find(3)
downtag=facet_tags.find(4)
obstacletag=facet_tags.find(5)

fluid_tag=cell_tags.find(7)


'''
with XDMFFile(MPI.COMM_WORLD, "Navie_domain.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

'''

t = 0.0
T = 8.0  # Final time
dt = 1 / 1600  # Time step size
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))  # Density



