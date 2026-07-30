from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from ufl import (
    Circumradius,
    FacetNormal,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
    div,
    dx,
    ds,
    grad,
    inner,
)

from dolfinx.fem import Function,functionspace,assemble_scalar,form
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio

domain=mesh.create_unit_square(MPI.COMM_WORLD,8,8)

tdim=domain.topology.dim
fdim=tdim-1



for i in [1,2,3]:
    V=functionspace(domain,("Lagrange",i))
    uex=Function(V)
    x=SpatialCoordinate(domain)
    uexx=1+x[0]**2+2*x[1]**2
    uex.interpolate(fem.Expression(uexx,V.element.interpolation_points))
    f=-div(grad(uex))

    u=TrialFunction(V)
    v=TestFunction(V)

    u_nitche=TrialFunction(V)

    # non nitch methode
    domain.topology.create_connectivity(fdim,tdim)
    boundaryfacet=mesh.exterior_facet_indices(domain.topology)
    boundarydofs=fem.locate_dofs_topological(V,fdim,boundaryfacet)
    bc=fem.dirichletbc(uex,boundarydofs)


    a=inner(grad(u),grad(v))*dx
    L=v*f*dx
    problem=LinearProblem(a,L,bcs=[bc],petsc_options_prefix="poisson")
    uh=problem.solve()

    er=form((uh-uex)**2*dx)
    err=np.sqrt(assemble_scalar(er))

    print(f'{i} degrees nonnitche error {err}')

    # nitche methode
    alp=fem.Constant(domain,default_scalar_type(10*i**2))
    n=FacetNormal(domain)
    h=2*Circumradius(domain)

    a_nit=inner(grad(u_nitche),grad(v))*dx
    a_nit-=inner(n,grad(u_nitche))*v*ds
    a_nit-=inner(n,grad(v))*u_nitche*ds
    a_nit+=(alp/h)*u_nitche*v*ds

    L_nit=v*f*dx
    L_nit-=inner(n,grad(v))*uex*ds
    L_nit+=(alp/h)*inner(uex,v)*ds

    problem_nit=LinearProblem(a_nit,L_nit,petsc_options_prefix="nitsche_poisson")
    uh_nit=problem_nit.solve()

    ern=form((uh_nit-uex)**2*dx)
    err_nit=np.sqrt(assemble_scalar(ern))

    print(f'{i} degrees nitche error {err_nit}')



    if i==5:
        with VTXWriter(domain.comm, "/home/ss/Capstone/poisson.bp", [uh]) as vtx:
            vtx.write(0.0)
        with VTXWriter(domain.comm, "/home/ss/Capstone/poisson_nit.bp", [uh_nit]) as vtx:
            vtx.write(0.0)


import numpy as np
import matplotlib.pyplot as plt

# 1 × 1 도메인
x = np.linspace(0, 1, 300)
y = np.linspace(0, 1, 300)

X, Y = np.meshgrid(x, y)

# 함수
U = 1 + X**2 + 2 * Y**2

# 등고선 채우기
plt.figure(figsize=(6, 5))

contour_filled = plt.contourf(
    X,
    Y,
    U,
    levels=20
)

# 등고선
contour_lines = plt.contour(
    X,
    Y,
    U,
    levels=20,
    colors="black",
    linewidths=0.7
)

plt.clabel(
    contour_lines,
    inline=True,
    fontsize=8
)

plt.colorbar(
    contour_filled,
    label=r"$u(x,y)=1+x^2+2y^2$"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$u(x,y)=1+x^2+2y^2$")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal")

plt.tight_layout()
plt.show()