
# FEniCSx Diffusion Tutorial - Modified Implementation

This code was written based on Jørgen S. Dokken's FEniCSx tutorial,  
**"Diffusion of a Gaussian function."**

The main purpose of this repository is not to reproduce the tutorial code exactly, but to adapt it to the FEniCSx/PETSc environment used in this project.

Because the DOLFINx API has changed across versions, several parts of the implementation differ from the tutorial.

This README focuses mainly on those differences.

---

# 1. Main Differences at a Glance

| Part | FEniCSx Tutorial | This Code | Main Reason / Effect |
|---|---|---|---|
| Scalar type | `PETSc.ScalarType(...)` | `dolfinx.default_scalar_type(...)` | Different scalar-type handling |
| Time step | Python scalar `dt` | `fem.Constant` for `dt` | `dt` becomes a UFL coefficient |
| Rectangle mesh | Explicit `mesh.CellType.triangle` | Cell type omitted | Uses default cell type |
| Boundary dimension | `domain.topology.dim - 1` | Hard-coded `1` | Simpler, but only appropriate here for 2D |
| Boundary value | Scalar value | `fem.Constant` | Different Dirichlet BC construction |
| RHS vector | `create_vector(...)` | `A.createVecRight()` | Workaround for vector API differences |
| Output | `XDMFFile` | `VTXWriter` | ADIOS2/BP4 output instead of XDMF |
| Initial output | Written at `t = 0` | Not written separately | First stored field is already after one solve |
| Time update | `t += dt` before solve/output | `t = t_start + dt_ss*i` | Current code has a one-step output-time offset |
| PyVista | Used for GIF/visualization | Removed | ParaView output only |
| PETSc cleanup | Explicit `destroy()` | Not used | PETSc objects left to Python cleanup |
| Output path | Relative path | Absolute path | Machine-specific output location |

Some of these changes are related to API/version differences, while others are simply implementation choices.

---

# 2. Scalar Type: `PETSc.ScalarType` vs `dolfinx.default_scalar_type`

The tutorial uses PETSc to define scalar values.

For example:

```python
PETSc.ScalarType(0)
```

The forcing term in the tutorial is also constructed using a PETSc scalar:

```python
f = fem.Constant(domain, PETSc.ScalarType(0))
```

In this code, the equivalent values are constructed using

```python
dolfinx.default_scalar_type(0)
```

For example:

```python
ff = fem.Constant(
    domain,
    dolfinx.default_scalar_type(0)
)
```

and

```python
bc = fem.dirichletbc(
    fem.Constant(
        domain,
        dolfinx.default_scalar_type(0)
    ),
    facetdof,
    V
)
```

Therefore, this code imports the root `dolfinx` module:

```python
import dolfinx
```

which is not required for the same purpose in the tutorial.

The intention is the same: create scalar data using the scalar type expected by the installed DOLFINx build.

This matters because DOLFINx can be compiled using different scalar types.

---

# 3. Dirichlet Boundary Condition Construction

The tutorial constructs the boundary condition approximately as

```python
bc = fem.dirichletbc(
    PETSc.ScalarType(0),
    fem.locate_dofs_topological(V, fdim, boundary_facets),
    V
)
```

In this code, a `fem.Constant` is created first:

```python
bc = fem.dirichletbc(
    fem.Constant(
        domain,
        dolfinx.default_scalar_type(0)
    ),
    facetdof,
    V
)
```

Therefore, the two implementations differ not only in

```python
PETSc.ScalarType
```

versus

```python
dolfinx.default_scalar_type
```

but also in the type of object supplied to `dirichletbc()`.

The tutorial passes a scalar value directly.

This implementation passes a DOLFINx `Constant`.

---

# 4. Boundary Dimension

The tutorial determines the facet dimension from the mesh:

```python
fdim = domain.topology.dim - 1
```

and then uses

```python
boundary_facets = mesh.locate_entities_boundary(
    domain,
    fdim,
    ...
)
```

This code instead directly uses

```python
facet = mesh.locate_entities_boundary(
    domain,
    1,
    lambda x: np.full(x.shape[1], True, dtype=bool)
)
```

and

```python
facetdof = fem.locate_dofs_topological(
    V,
    1,
    facet
)
```

For this 2D problem,

```text
domain dimension = 2
facet dimension  = 1
```

so both implementations select the same type of entity.

However, the tutorial implementation is more general.

For example, if the mesh were changed to 3D, the boundary facets would have dimension

```text
3 - 1 = 2
```

and the tutorial code would continue to work.

The hard-coded value

```python
1
```

would not.

Therefore, this difference is **not mainly a version difference**.

It is a simplification specific to the current 2D problem.

---

# 5. Mesh Creation

The tutorial explicitly specifies triangular cells:

```python
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-2, -2]), np.array([2, 2])],
    [nx, ny],
    mesh.CellType.triangle
)
```

This implementation uses

```python
l = np.array([-2, -2])
ll = np.array([2, 2])

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [l, ll],
    [50, 50]
)
```

The main difference is that

```python
mesh.CellType.triangle
```

is not explicitly supplied.

The current DOLFINx API uses triangular cells as the default for `create_rectangle()`, so the resulting cell type is still triangular.

However, the tutorial explicitly states the intended cell type while this implementation relies on the default argument.

Therefore,

```python
mesh.CellType.triangle
```

is not necessary here, but explicitly writing it makes the mesh definition clearer.

---

# 6. Initial-Condition Function Naming

The tutorial uses

```python
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))
```

This code uses

```python
def f(x):
    return np.exp(-5 * (x[0]**2 + x[1]**2))
```

The numerical function is essentially the same.

However, the tutorial later uses the name

```python
f
```

for the source term.

Therefore, the tutorial has separate names:

```text
initial_condition
f
```

This code instead uses

```text
f   -> initial condition
ff  -> source term
```

as shown below:

```python
def f(x):
    return np.exp(-5 * (x[0]**2 + x[1]**2))

ff = fem.Constant(
    domain,
    dolfinx.default_scalar_type(0)
)
```

This is only a naming difference, not a numerical difference.

---

# 7. `dt`: Python Scalar vs `fem.Constant`

This is another important difference.

The tutorial defines the time step as a normal Python number:

```python
dt = T / num_steps
```

and directly uses it in the UFL form:

```python
a = u * v * ufl.dx \
    + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
```

This implementation first calculates a Python scalar:

```python
dt_ss = (t_end - t_start) / steps
```

and then converts it into a DOLFINx `Constant`:

```python
dt = fem.Constant(
    domain,
    dolfinx.default_scalar_type(dt_ss)
)
```

Thus there are actually two representations of the time step:

```text
dt_ss -> ordinary Python/NumPy scalar
dt    -> DOLFINx Constant used in the weak form
```

The loop uses `dt_ss`:

```python
t = t_start + dt_ss * i
```

while the variational form uses `dt`:

```python
a = (
    u * v * ufl.dx
    + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
)

L = (u_n + dt * ff) * v * ufl.dx
```

Using a `fem.Constant` is useful if the value is intended to be changed later without reconstructing the UFL expression.

In the current code, however, `dt` is constant during the entire simulation, so using `fem.Constant` is not strictly necessary.

This is therefore mainly an implementation choice rather than a required version change.

---

# 8. The Most Important API Difference: RHS Vector Creation

This is probably the most significant difference between the tutorial and this implementation.

The current tutorial creates the RHS vector using DOLFINx:

```python
b = create_vector(
    fem.extract_function_spaces(linear_form)
)
```

The important point is that recent DOLFINx versions define `create_vector()` in terms of the associated function space.

Older DOLFINx APIs used different forms of `create_vector()`, and code written for one version may therefore fail in another version.

This implementation does not use the imported DOLFINx function at all.

Although

```python
create_vector
```

is imported,

```python
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
```

the actual vector is created using PETSc:

```python
A = assemble_matrix(a_form, bcs=[bc])
A.assemble()

b = A.createVecRight()
```

Therefore,

```python
create_vector
```

is currently an unused import.

---

# 9. Why `A.createVecRight()` Works Here

PETSc matrices can generate vectors compatible with their matrix dimensions.

Therefore,

```python
b = A.createVecRight()
```

creates a vector compatible with the right side of the matrix.

For the linear system

```text
A u = b
```

the vector has the correct global size for the unknown represented by the columns of `A`.

This is why it can be used successfully in the present single-process calculation.

The approach also avoids depending on the exact DOLFINx `create_vector()` calling convention.

However, this is not exactly the same abstraction as the DOLFINx tutorial uses.

The tutorial creates `b` from the finite-element function-space layout.

This implementation creates `b` from the PETSc matrix layout.

That distinction becomes more important in parallel calculations.

---

# 10. Important MPI Difference: Ghosted Vector

DOLFINx finite-element vectors contain information about owned and ghost degrees of freedom in distributed MPI calculations.

The tutorial therefore deliberately creates a DOLFINx-compatible PETSc vector.

The tutorial then performs

```python
b.ghostUpdate(
    addv=PETSc.InsertMode.ADD_VALUES,
    mode=PETSc.ScatterMode.REVERSE
)
```

after assembly.

This code performs exactly the same operation:

```python
b.ghostUpdate(
    addv=PETSc.InsertMode.ADD_VALUES,
    mode=PETSc.ScatterMode.REVERSE
)
```

but `b` was created using

```python
A.createVecRight()
```

instead of DOLFINx's `create_vector()`.

This distinction is not important when the program is executed using only one MPI rank because there are no inter-process ghost degrees of freedom to communicate.

However, `A.createVecRight()` should not automatically be regarded as a general MPI replacement for DOLFINx's `create_vector()`.

For parallel FEniCSx code, the DOLFINx-created ghosted vector is preferable because its layout is explicitly constructed from the finite-element function space and its index map.

In other words:

```text
Single process:
A.createVecRight()
    -> sufficient for this code

Multiple MPI processes:
DOLFINx create_vector(...)
    -> safer and consistent with the FEniCSx assembly model
```

This is particularly relevant because the current code contains the comment

```python
# 어차피 단일코어로 할건데
```

and is effectively written with single-process execution in mind.

---

# 11. Matrix Assembly Is Almost Identical to the Tutorial

The matrix assembly itself has not substantially changed.

Tutorial:

```python
bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(
    bilinear_form,
    bcs=[bc]
)

A.assemble()
```

This implementation:

```python
a_form = fem.form(a)
L_form = fem.form(L)

A = assemble_matrix(
    a_form,
    bcs=[bc]
)

A.assemble()
```

The main difference is only variable naming:

```text
bilinear_form -> a_form
linear_form   -> L_form
```

The numerical procedure is the same.

The matrix is assembled only once because the left-hand side does not change during the time loop.

---

# 12. PETSc Solver Is Essentially the Same

The tutorial creates the solver using

```python
solver = PETSc.KSP().create(domain.comm)

solver.setOperators(A)

solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
```

This implementation uses exactly the same solver configuration:

```python
solver = PETSc.KSP().create(domain.comm)

solver.setOperators(A)

solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
```

Therefore, this part is not a version-related modification.

Both use

```text
KSP = PREONLY
PC  = LU
```

which means that the PETSc preconditioner performs the actual direct solve.

This is one of the parts that was retained almost unchanged from the tutorial.

---

# 13. RHS Assembly Is Also Almost the Same

The tutorial resets and reassembles the RHS vector every time step:

```python
with b.localForm() as loc_b:
    loc_b.set(0)

assemble_vector(b, linear_form)
```

This implementation does the same:

```python
with b.localForm() as loc_b:
    loc_b.set(0)

assemble_vector(b, L_form)
```

The boundary-condition procedure is also preserved:

```python
apply_lifting(b, [a_form], [[bc]])

b.ghostUpdate(
    addv=PETSc.InsertMode.ADD_VALUES,
    mode=PETSc.ScatterMode.REVERSE
)

set_bc(b, [bc])
```

Therefore, the following DOLFINx assembly sequence was retained from the tutorial:

```text
reset RHS
    ↓
assemble_vector
    ↓
apply_lifting
    ↓
ghostUpdate
    ↓
set_bc
    ↓
solve
```

The major difference in this part is not the assembly algorithm itself.

It is how the vector `b` was originally created.

---

# 14. XDMF Output Was Replaced by `VTXWriter`

The tutorial uses

```python
io.XDMFFile
```

for time-dependent output.

It first creates the XDMF file:

```python
xdmf = io.XDMFFile(
    domain.comm,
    "diffusion.xdmf",
    "w"
)
```

and explicitly writes the mesh:

```python
xdmf.write_mesh(domain)
```

The initial solution is then written using

```python
xdmf.write_function(uh, t)
```

and later time steps are appended to the same output.

This implementation instead uses

```python
from dolfinx.io import VTXWriter
```

and

```python
vtx_uh = VTXWriter(
    domain.comm,
    "/home/ss/Capstone/Tuto_hit.bp",
    [uh],
    engine="BP4"
)
```

The result is then written with

```python
vtx_uh.write(t)
```

Therefore, the output format changed from

```text
XDMF / HDF5
```

to

```text
VTX / ADIOS2 BP4
```

This is one of the largest practical changes from the tutorial.

The resulting `.bp` data can be opened with ParaView.

---

# 15. The Output Path Is Machine-Specific

The tutorial uses a relative output path such as

```python
"diffusion.xdmf"
```

This implementation uses

```python
"/home/ss/Capstone/Tuto_hit.bp"
```

which is an absolute path.

This means the current script depends on the directory structure of the machine on which it was written.

For a GitHub repository, a relative path would generally be more portable.

For example:

```python
"Tuto_hit.bp"
```

or

```python
"results/Tuto_hit.bp"
```

would make the script easier to run on another computer.

This is unrelated to the FEniCSx version.

---

# 16. PyVista Visualization Was Removed

The tutorial performs two types of output:

```text
1. XDMF output for ParaView
2. PyVista visualization / GIF generation
```

It creates a PyVista grid and plotter and updates the visualization during the time loop.

This implementation imports

```python
import matplotlib as mpl
import pyvista
```

and also imports

```python
from dolfinx import fem, mesh, io, plot
```

but the PyVista plotting section itself is not present.

Therefore,

```python
mpl
pyvista
io
plot
```

are currently unused imports.

The script only produces the `.bp` output for external visualization.

This makes the implementation substantially simpler than the complete tutorial example.

---

# 17. `Path` Is Also Currently Unused

The script contains

```python
from pathlib import Path
```

and previously appears to have intended to create a result directory:

```python
# folder = Path("results")
# folder.mkdir(exist_ok=True)
```

Since those lines are commented out, `Path` is currently unused.

Therefore, the following import could also be removed:

```python
from pathlib import Path
```

unless result-directory creation is restored later.

---

# 18. Initial Output Is Different

The tutorial initializes `uh`:

```python
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
```

and immediately writes it at the initial time:

```python
xdmf.write_function(uh, t)
```

At this point,

```text
t = 0
```

so the output contains the actual initial condition.

This implementation also initializes `uh`:

```python
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(f)
```

but does not immediately write it.

The first

```python
vtx_uh.write(t)
```

occurs only after the first linear solve.

Therefore, the `.bp` output does not contain a separately written initial-condition state.

---

# 19. Important Difference: Current Time Label Is Shifted by One Step

This is a behavioral difference that should be noted carefully.

The tutorial starts with

```python
t = 0.0
```

and writes the initial field at

```text
t = 0
```

Then, inside the loop, it performs

```python
t += dt
```

before writing the newly calculated solution.

Therefore the sequence is

```text
initial condition -> t = 0.00

first solve       -> t = 0.02
second solve      -> t = 0.04
...
50th solve        -> t = 1.00
```

The current implementation instead uses

```python
for i in range(steps):
    t = t_start + dt_ss * i
```

before solving.

With

```text
dt = 0.02
```

this produces

```text
i = 0  -> t = 0.00
i = 1  -> t = 0.02
...
i = 49 -> t = 0.98
```

However, `vtx_uh.write(t)` occurs **after** the solve.

Therefore, the first calculated solution, which physically corresponds to the first time step, is written with the label

```text
t = 0.00
```

instead of

```text
t = 0.02
```

and the final solution after 50 steps is written as

```text
t = 0.98
```

instead of

```text
t = 1.00
```

So the current output time is shifted backward by one time step.

A loop consistent with the tutorial would be, for example:

```python
vtx_uh.write(t_start)

for i in range(steps):
    t = t_start + (i + 1) * dt_ss

    with b.localForm() as loc_b:
        loc_b.set(0)

    assemble_vector(b, L_form)

    apply_lifting(b, [a_form], [[bc]])

    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE
    )

    set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_n.x.array[:] = uh.x.array

    vtx_uh.write(t)
```

This would give

```text
initial condition -> 0.00
first solution    -> 0.02
...
final solution    -> 1.00
```

This issue is not caused by a DOLFINx version difference.

It comes from changing the tutorial's time-update logic.

---

# 20. Explicit PETSc Object Destruction Was Removed

At the end of the tutorial, the PETSc objects are explicitly destroyed:

```python
A.destroy()
b.destroy()
solver.destroy()
```

This implementation only closes the output writer:

```python
vtx_uh.close()
```

and does not explicitly call

```python
A.destroy()
b.destroy()
solver.destroy()
```

For a short script this will usually not be noticeable because the Python process exits immediately afterwards.

However, the tutorial explicitly releases the PETSc resources.

This difference may become more relevant in a larger program in which many PETSc matrices, vectors, or solvers are repeatedly created.

---

# 21. `VTXWriter` Is Explicitly Closed

Although PETSc objects are not explicitly destroyed, the VTX writer is correctly closed:

```python
vtx_uh.close()
```

This plays a similar role to

```python
xdmf.close()
```

in the tutorial.

So the output resource itself is explicitly finalized.

---

# 22. `create_vector` Import Is Left Over

The code imports

```python
create_vector
```

here:

```python
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
```

but the actual vector is constructed with

```python
b = A.createVecRight()
```

Therefore, `create_vector` is no longer used.

This appears to be a remnant of adapting the original tutorial implementation to a different API/environment.

It can be removed if `A.createVecRight()` continues to be used.

---

# 23. `assemble_vector` Creation Style Also Changed

The code contains the commented line

```python
# assemble_vector(b, L_form)
```

immediately after

```python
b = A.createVecRight()
```

but the actual assembly is delayed until the time loop.

This follows the correct idea used by the tutorial:

```text
Create vector once
Assemble its values repeatedly
```

The important distinction is between

```text
vector creation
```

and

```text
vector assembly
```

The vector object `b` is created once.

Its numerical values are then reset and assembled at every time step.

---

# 24. `LinearProblem` Is Not Used

The script contains

```python
# problem=PETSc.LinearProblem(a,L,bcs=[bc],petsc_options_prefix="")
```

but the actual implementation uses manual matrix/vector assembly and a PETSc KSP solver.

This is consistent with the main structure of the Dokken tutorial example.

Once the matrix is explicitly assembled and reused,

```python
A = assemble_matrix(...)
```

the low-level PETSc approach makes it possible to control

```text
matrix assembly
RHS assembly
boundary lifting
ghost communication
solver configuration
```

individually.

Therefore, this code should be understood as an explicitly assembled PETSc workflow rather than a high-level `LinearProblem` workflow.

---

# 25. What Was Actually Preserved from the Tutorial

Despite the API and output differences, the central FEniCSx workflow remains very similar.

The following operations are essentially retained from the tutorial:

```text
Create mesh
    ↓
Create function space
    ↓
Interpolate initial condition
    ↓
Find boundary facets
    ↓
Create Dirichlet boundary condition
    ↓
Create UFL trial/test functions
    ↓
Create bilinear and linear forms
    ↓
Compile forms with fem.form()
    ↓
Assemble A only once
    ↓
Create PETSc KSP solver
    ↓
For each time step:
    reset b
    assemble b
    apply lifting
    reverse ghost update
    impose BC
    solve
    forward scatter
    update u_n
    write result
```

Therefore, the main numerical architecture of the tutorial is still present.

The biggest changes occur around the API used to represent scalars, create the RHS vector, and write output data.

---

# 26. Which Differences Are Probably Version-Related?

The differences should not all be attributed to the DOLFINx version.

The most version-sensitive parts are:

### Scalar handling

Tutorial:

```python
PETSc.ScalarType(...)
```

This implementation:

```python
dolfinx.default_scalar_type(...)
```

### RHS vector construction

Tutorial:

```python
create_vector(
    fem.extract_function_spaces(linear_form)
)
```

This implementation:

```python
A.createVecRight()
```

The API surrounding vector creation has changed between DOLFINx releases, so this is the part most likely to cause errors when tutorial code is copied directly into another installed version.

### Output API

The tutorial uses

```python
XDMFFile
```

while the implementation uses

```python
VTXWriter
```

Both are valid output strategies, but their availability and preferred usage have also evolved with DOLFINx and ADIOS2 support.

---

# 27. Which Differences Are NOT Version-Related?

The following changes are mainly choices made in this implementation:

```text
Hard-coded facet dimension = 1
Using fem.Constant for dt
Removing PyVista visualization
Using an absolute output path
Renaming the source term to ff
Omitting u_n.name
Omitting explicit PETSc destroy()
Omitting the explicit initial output
Changing the time-update expression
```

These should not be described simply as consequences of a newer DOLFINx version.

---

# 28. Important Note for Parallel Execution

The current implementation was written mainly for single-process execution.

In particular,

```python
b = A.createVecRight()
```

should be treated carefully before running the program with

```bash
mpirun -n 4 python3 Tutorial_Heat.py
```

or another multi-process configuration.

The tutorial creates the RHS vector using the DOLFINx function-space/index-map information, which provides the ghost structure needed for distributed finite-element assembly.

Therefore, when converting this example to a genuinely parallel simulation, it is better to return to the DOLFINx-compatible vector-creation API appropriate for the installed DOLFINx version.

For the API used by the current tutorial, that is written as

```python
b = create_vector(
    fem.extract_function_spaces(L_form)
)
```

rather than

```python
b = A.createVecRight()
```

The exact call should still be checked against the installed DOLFINx version.

---

# 29. Checking the Installed Version

Because DOLFINx changes relatively quickly, the exact version should be recorded when reproducing this code.

It can be checked with

```python
print("DOLFINx:", dolfinx.__version__)
print("PETSc:", PETSc.Sys.getVersion())
```

or from the terminal with the package-management tool used to install FEniCSx.

Recording the version is useful because code examples found in older FEniCSx tutorials may use different forms of

```text
create_vector
LinearProblem
dirichletbc
FunctionSpace/functionspace
PETSc vector access
output writers
```

even when the underlying finite-element formulation is unchanged.

---

# 30. Summary

The implementation is based closely on the Dokken diffusion tutorial, but it is not a direct copy.

The most important modifications are:

1. `PETSc.ScalarType` was replaced by `dolfinx.default_scalar_type`.

2. The Dirichlet value is represented using `fem.Constant`.

3. `dt` is represented as a `fem.Constant` instead of only a Python scalar.

4. The tutorial's DOLFINx RHS-vector creation was replaced by

   ```python
   b = A.createVecRight()
   ```

   mainly to avoid the version-dependent `create_vector()` interface.

5. The core PETSc assembly sequence

   ```text
   assemble_vector
   -> apply_lifting
   -> ghostUpdate
   -> set_bc
   -> solve
   ```

   remains almost identical to the tutorial.

6. XDMF output was replaced by `VTXWriter` with ADIOS2 BP4 output.

7. PyVista/GIF visualization was removed.

8. Some tutorial imports are consequently unused.

9. The present `A.createVecRight()` approach is suitable for the intended single-process use, but should not automatically be assumed to replace a DOLFINx ghosted vector in MPI calculations.

10. The current output time labels are shifted by one time step because the time-update logic differs from the tutorial.

Thus, most of the finite-element assembly procedure remains the same, while the main differences lie in **DOLFINx/PETSc API compatibility, vector creation, scalar representation, and result output**.
````
