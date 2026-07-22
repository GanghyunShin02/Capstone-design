import dolfinx
from mpi4py import MPI
import gmsh
from dolfinx import mesh,io
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio
import numpy as np

gmsh.initialize()

disk=gmsh.model.occ.addDisk(0,0,0,1,1)
eye=gmsh.model.occ.addDisk(0,2/3,0,0.1,0.1)

# Sector create

#중점
dot_centor=gmsh.model.occ.addPoint(0,0,0)


x=(2**0.5)/2
dot1=gmsh.model.occ.addPoint(x,x,0)
dot2=gmsh.model.occ.addPoint(x,-x,0)
dot3=gmsh.model.occ.addPoint(1,0,0)

l1=gmsh.model.occ.addLine(dot_centor,dot1)
l2=gmsh.model.occ.addLine(dot_centor,dot2)
curve=gmsh.model.occ.addCircleArc(dot1,dot_centor,dot2)

loop=gmsh.model.occ.addCurveLoop([l1,curve,l2])
surf=gmsh.model.occ.addPlaneSurface([loop])


packman,_=gmsh.model.occ.cut([(2,disk)],[(2,surf),(2,eye)])


gmsh.model.occ.synchronize()

x1=(2**0.5)/4

#눈부분 조밀하게
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",30)


# 입부분조밀하게
gmsh.model.mesh.field.add("Box",1)
gmsh.model.mesh.field.setNumber(1,"VIn",0.01)
gmsh.model.mesh.field.setNumber(1,"VOut",0.1)
gmsh.model.mesh.field.setNumber(1,"XMin",-x1)
gmsh.model.mesh.field.setNumber(1,"XMax",x1)
gmsh.model.mesh.field.setNumber(1,"YMax",x1)
gmsh.model.mesh.field.setNumber(1,"YMin",-x1)
gmsh.model.mesh.field.setNumber(1,"ZMin",0)
gmsh.model.mesh.field.setNumber(1,"ZMax",0)
#gmsh.model.mesh.field.setNumber(1,"Thickness",0.2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

gmsh.model.mesh.field.setAsBackgroundMesh(1)



tags=[tags for dim,tags in packman]

gmsh.model.addPhysicalGroup(2,tags,2)

gmsh.model.mesh.generate(2)

meshdata=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,2)
domain=meshdata.mesh


gmsh.finalize()

print('f')


with XDMFFile(MPI.COMM_WORLD, "packman6.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)


