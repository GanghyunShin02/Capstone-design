
import dolfinx
import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

#  어차ㅠㅣ 단일코어로 할건데

l=np.array([-2,-2])
ll=np.array([2,2])

domain=mesh.create_rectangle(MPI.COMM_WORLD,[l,ll],[50,50])

V=fem.functionspace(domain,("Lagrange",1))

u_n=fem.Function(V)

def f(x):
    return np.exp(-5*(x[0]**2+x[1]**2))

u_n.interpolate(f)



facet=mesh.locate_entities_boundary(domain,1,lambda x: np.full(x.shape[1],True,dtype=bool))
facetdof=fem.locate_dofs_topological(V,1,facet)
bc=fem.dirichletbc(fem.Constant(domain,dolfinx.default_scalar_type(0)),facetdof,V)


t_start,t_end=0,1
steps=50
dt_ss=(t_end-t_start)/steps
dt=fem.Constant(domain,dolfinx.default_scalar_type(dt_ss))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
ff = fem.Constant(domain, dolfinx.default_scalar_type(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * ff) * v * ufl.dx

#problem=PETSc.LinearProblem(a,L,bcs=[bc],petsc_options_prefix="")

a_form=fem.form(a)
L_form=fem.form(L)

A=assemble_matrix(a_form,bcs=[bc])
A.assemble()
b=A.createVecRight()
#assemble_vector(b,L_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


uh=fem.Function(V)
uh.name='uh'
uh.interpolate(f)

from pathlib import Path
from dolfinx.io import VTXWriter
#folder = Path("results")
#folder.mkdir(exist_ok=True)

vtx_uh = VTXWriter(domain.comm, "/home/ss/Capstone/Tuto_hit.bp", [uh], engine="BP4")



for i in range(steps):
    t=t_start+dt_ss*i

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L_form)

    print(f'after assemble vec')

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])


    print('after aplylifting')
    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    vtx_uh.write(t)


vtx_uh.close()








