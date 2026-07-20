import dolfinx
from mpi4py import MPI
import gmsh
from dolfinx import mesh,io
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio
import numpy as np

'''
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
)
'''




import gmsh
'''
gmsh.initialize()
# Create geometry
disk=gmsh.model.occ.addDisk(0, 0, 0, 1, 1)  # unit disk
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2,[disk],1)

# Set mesh size
gmsh.model.mesh.setSize(
    gmsh.model.getEntities(0), 0.1
)
# Generate 2D mesh
gmsh.model.mesh.generate(2)
# Import into DOLFINx

meshdata=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,2)
dom=meshdata.mesh

cell_tags=meshdata.cell_tags
facet_tags=meshdata.facet_tags

gmsh.finalize()

'''

'''
meshdata1=gmshio.read_from_msh(
    "/home/ss/untitled.msh",MPI.COMM_WORLD,
    0,3
)

dom1=meshdata1.mesh
cell_tags1=meshdata1.cell_tags
facet_tags1=meshdata1.facet_tags


with XDMFFile(MPI.COMM_WORLD, "dom1.xdmf", "w") as xdmf:
    xdmf.write_mesh(dom1)
'''

'''
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

with XDMFFile(MPI.COMM_WORLD, "msh.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)


'''




'''
with XDMFFile(MPI.COMM_WORLD, "tri.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_tri)


with XDMFFile(MPI.COMM_WORLD, "quad.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_quad)

with XDMFFile(MPI.COMM_WORLD, "rect.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_rect)
with XDMFFile(MPI.COMM_WORLD, "cube.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_3d)
with XDMFFile(MPI.COMM_WORLD, "right.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_right)
with XDMFFile(MPI.COMM_WORLD, "left.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_left)
with XDMFFile(MPI.COMM_WORLD, "cross.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_crossed)

'''


'''
square=mesh.create_unit_square(MPI.COMM_WORLD,
                               4,4,
                               diagonal=mesh.DiagonalType.right)

tdim=square.topology.dim

for dim in range(0,tdim+1):
    square.topology.create_entities(dim)

square.topology.create_connectivity(tdim, 0)
square.topology.create_connectivity(tdim, tdim - 1)

for d in range(tdim + 1):
    n = square.topology.index_map(d).size_local
    names = {0: "vertices", 1: "edges", 2: "cells"}
    print(f"  dim {d} ({names[d]}): {n}")

vertic=square.topology.connectivity(tdim,0)

print(f'Return all cells verticle index{vertic}')
#print(type(vertic))
'''

'''
<class 'dolfinx.cpp.graph.AdjacencyList_int32'>
Because type is special,
call like `vertic.links(n)`.
'''

'''
coords=square.geometry.x
b=[]
detJ=[]

for i in range(square.topology.index_map(tdim).size_local):

    a=[int(x)  for x in vertic.links(i) ]# i번 셀의 정점인덱스를 저장
    k=(coords[a])
    b.append(k)

    J1=-k[0][0]+k[1][0]
    J2=-k[0][0]+k[2][0]
    J3=-k[0][1]+k[1][1]
    J4=-k[0][1]+k[2][1]

    dJ=J1*J4-J2*J3
    detJ.append(dJ)


#print(b)

print(f'Return cell(0)s verticle Index{vertic.links(0)}')

print(f'Return cell(0)s Real coordinate{coords[vertic.links(0)]}')
print(detJ)
'''


'''

mesh11=mesh.create_unit_square(MPI.COMM_WORLD,3,3,
                               )

with XDMFFile(MPI.COMM_WORLD, "exercise2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh11)


mesh11.topology.create_connectivity(2,0)
mesh11.topology.create_connectivity(2,1)
mesh11.topology.create_connectivity(1,0)

Adj20=mesh11.topology.connectivity(2,0)
Adj21=mesh11.topology.connectivity(2,1)
Adj10=mesh11.topology.connectivity(1,0)
#Adj12=mesh11.topology.connectivity(1,2)

print(f'$Adj_{{2,0}}${Adj20}')
print(f'$Adj_{{2,1}}${Adj21}')
print(f'$Adj_{{1,0}}${Adj10}')
#print(f'$Adj_{1,2}${Adj12}')


cnt=0

for i in range(mesh11.topology.index_map(1).size_local):

    for j in range(mesh11.topology.index_map(2).size_local):
        x=Adj21.links(j)
        if x[0]==i or x[1]==i or x[2]==i :
            cnt+=1
            print(f'Edge {i} in cell{j}')

        if cnt==3:
            print('broken mesh')
            break

'''

gmsh.initialize()

a=gmsh.model.occ.addRectangle(0,0,0,1,1)
b=gmsh.model.occ.addRectangle(0.5,0.5,0,0.5,0.5)
c,_=gmsh.model.occ.cut([(2,a)],[(2,b)])

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2,[c[0][1]],1,'s')

tag=gmsh.model.getBoundary(c,oriented=False)

for dim,tags in tag:
    L=gmsh.model.occ.getMass(dim,tags)
    if round(L)==1:
        gmsh.model.mesh.setTransfiniteCurve(tags,3)
    else:
        gmsh.model.mesh.setTransfiniteCurve(tags,2)



gmsh.model.mesh.generate(2)

meshdata=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,2)
domain=meshdata.mesh

gmsh.finalize()

with XDMFFile(MPI.COMM_WORLD, "Lshape4.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)



domain.topology.create_connectivity(2,1)
domain.topology.create_connectivity(1,0)


print(f'vertex{domain.topology.index_map(0).size_local}')
print(f'edgy{domain.topology.index_map(1).size_local}')
print(f'cell {domain.topology.index_map(2).size_local}')


Adj21=domain.topology.connectivity(2,1)

y=0
z=0

for i in range(domain.topology.index_map(1).size_local):
    for j in range(domain.topology.index_map(2).size_local):
        x=Adj21.links(j)
        if x[0]==i or x[1]==i or x[2]==i:
            y+=1
    if y==1: z+=1
    y=0

print(f'boundary facet {z}')







