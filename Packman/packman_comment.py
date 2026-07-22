#==  기본 import.. 이러이러한 모듈..패키지를 쓰겠다는 선언입니다..==
# 파이선은 다른사람들이 다른 컴퓨팅언어로 만든 ...프로그램들을 쓸수있습니다.
# FEniCSx 또한 여러 패키지중 하나로 볼수있습니다...
# FEniCSx 코딩할때 일단 임포트코드들을 붙여넣기하는게 편합니다 .

# dolfinx 를 쓰겠다는 선언입니다.
import dolfinx
# mpi 'for' 'python'
# === MPI ===
# message passing interface...
# 복잡한 계산은 CPU의 코어들을 병렬로 연결해서 수행합니다.
# 그 코어간의 통신에관한 인터페이스입니다.
# mpi 'for' 'python 에서 MPI를 import 
from mpi4py import MPI
# gmsh 를 import..
import gmsh
# 
from dolfinx import mesh,io
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio

import numpy as np

#gmsh  시작
gmsh.initialize()

#===원판생성===
#gmsh.model.occ.add...(x,y,z,dx,dy,dz)
#중점좌표(0,0,0).. 항상 삼차원 좌표를 사용
#(1,1) 장반경..단반경.. 그냥 반지름..
#2차원이므로 dx,dy만 줘도 가능.. 점은 항상 삼차원으로..

disk=gmsh.model.occ.addDisk(0,0,0,1,1)
eye=gmsh.model.occ.addDisk(0,2/3,0,0.1,0.1)


#중점
dot_centor=gmsh.model.occ.addPoint(0,0,0)

# cos(45).. 거듭제곱은 x**n 으로 표현...
x=(2**0.5)/2

#=== 생성한 원판에서 부채꼴을 빼야한다. 그 부채꼴을 만드는 과정
#그림참조..
dot1=gmsh.model.occ.addPoint(x,x,0)
dot2=gmsh.model.occ.addPoint(x,-x,0)
dot3=gmsh.model.occ.addPoint(1,0,0)

l1=gmsh.model.occ.addLine(dot_centor,dot1)
l2=gmsh.model.occ.addLine(dot_centor,dot2)
#addCircleArc로 원호를 추가.. 
# addCircleArc(끝점1,원의 중점,끝점2) 의 형식..
curve=gmsh.model.occ.addCircleArc(dot1,dot_centor,dot2)

#경계를 지정..선들의 집합..
loop=gmsh.model.occ.addCurveLoop([l1,curve,l2])
# 그 경계를 이루는 면을 선언..
surf=gmsh.model.occ.addPlaneSurface([loop])

#원판에서 부채꼴과 눈모양을 자른다.
#...cut([(d,name)])에서 d는 차원.
packman,_=gmsh.model.occ.cut([(2,disk)],[(2,surf),(2,eye)])

#===occ란...===
#Open CAS Cade.. 널리 쓰이는 오픈소스 형상 모델링 프로그램..
#occ로 만든 형상을 synchronize()함수로 gmsh.model에 가지고온다.
gmsh.model.occ.synchronize()

x1=(2**0.5)/4

#===눈부분 조밀하게===
#이렇게안하면 다각형이됨.
# "Mesh.MeshSizeFromCurvature" 
# 으로 굴곡진곳은 자동으로 30분할하라는 명령..
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",30)


#=== 입부분조밀하게===

# 조밀하게 만들 영역은 "Box".. 직사각형모양이고 그 영역의 이름은 1.
# tagging이라고하며.. 정수로 tag해야함.. 
gmsh.model.mesh.field.add("Box",1)
#"Box" 영역안의 메쉬크기는 0.01로 하겠다는 명령
gmsh.model.mesh.field.setNumber(1,"VIn",0.01)
# "Box"밖은 0.1로..
gmsh.model.mesh.field.setNumber(1,"VOut",0.1)
# Box 영역의 크기를 지정..
# x,y,z,최소-최대를 지정..
gmsh.model.mesh.field.setNumber(1,"XMin",-x1)
gmsh.model.mesh.field.setNumber(1,"XMax",x1)
gmsh.model.mesh.field.setNumber(1,"YMax",x1)
gmsh.model.mesh.field.setNumber(1,"YMin",-x1)
gmsh.model.mesh.field.setNumber(1,"ZMin",0)
gmsh.model.mesh.field.setNumber(1,"ZMax",0)
# Thickness 는 서로 크기가 다른 메쉬 사이를 자연스럽게 연결해주는..명령..
# 예를들어 크기 1인 메쉬옆에 0.1인 메쉬가 있다면.. 그 사이를 0.5정도로 잘라서
# 자연스레...? 만듦? 
# 이번 과제에서는 굳이 그럴필요없다고생각해서 주석처리함.
#gmsh.model.mesh.field.setNumber(1,"Thickness",0.2)

# gmsh 내부 알고리즘으로 내가 지정한모양에서 메쉬 크기가 바뀜..
# 그 기능을 끄겠다는 명령
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

#mesh.field.add("Box",1) 로.. 만든 1번장(field)를 사용한다는 명령
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# 이번 과제에서는 딱히 영역을 나누어서 계산할필요가없으므로 그냥 전체를 하나의 그룹으로묶음
# packman 이란 메쉬는 (차원,index) *index는 내부에서 갖고있는 이름.. tag는 내가 붙인 이름
# tags 란 배열에 packman의 index들을 저장하겠다는 코드
tags=[tags for dim,tags in packman]
# 그걸 그냥 피지컬그룹으로 묶음..
# gmsh로 모델을 만들때는 항상 피지컬그룹이필요함.. 이번에는 그냥 막 지정해준것.
gmsh.model.addPhysicalGroup(2,tags,2)

#gmsh로 모델을 만든다. 2차원의
gmsh.model.mesh.generate(2)

#gmsh.model 을 dolfinx로 갖고온다.
meshdata=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,2)
#그 메쉬이름을 domain 으로 지정...
domain=meshdata.mesh

# 끝...
gmsh.finalize()

# 코드가 잘돌아가는지 확인하기위한 디버깅코드입니다..
# 코드를 만들다가 생각한대로 잘 안돌아가면 
# 각 구간마다 이런식으로 출력문을 넣어서 어디서 막히는지 알수 있습니다.
print('f')

#===  시각화파일생성 ===
# paraview 로 시각화하기위해 xdmf파일을 생성합니다.
# "packman6.xdmf"란 파일이름으로 만들겠다는 명령입니다.
# "w" 는 wrigh v파일을 만들겠다는 명령입니다.
# 이 명령을 실행하면.. 이 파이선파일이 있는경로에 
# packman6.xdmf 와 packman6.h5 파일이 생성될겁니다.
# paraview에서 이 파일 두개를 열고.. 둘중하나를 apply하면 시각화됩니다.
# colab에서는..왼쪽에있는.. 폴더모양버튼울 누르면 파일이 보이실겁니다..
# 그 파일들을 다운로드해서 시각화하면됩니다.... 
with XDMFFile(MPI.COMM_WORLD, "packman6.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)


