explain the f-symbol and n-symbol storage system

# MultiTensorKit implementation: $\mathsf{Rep_{A_4}}$ as an example
This tutorial is dedicated to explaining how MultiTensorKit was implemented to be compatible with with TensorKit and MPSKit for matrix product state simulations. In particular, we will be making a generalised anyonic spin chain. We will demonstrate ... (not sure yet what we're going to show here). 

This will be done with the `A4Object = BimoduleSector{A4}` `Sector`, which is the multifusion category which contains the structure of the module categories over $\mathsf{Rep_{A_4}}$. Since there are 12 module categories (technically only 8 up to equivalence), `A4Object` is a $r=12$ multifusion category. There are 3 fusion categories up to equivalence:
- $\mathsf{Vec A_4}$: the category of $\mathsf{A_4}$-graded vector spaces. The group $\mathsf{A}_4$ is order $4!/2 = 12$. It has thus 12 objects.
- $\mathsf{Rep_{A_4}}$: the irreducible representations of the group $\mathsf{A}_4$, of which there are 4. One is the trivial representation, two are one-dimension non-trivial and the last is three-dimensional.
- $\mathsf{Rep H}$: the representation category of some Hopf algebra which does not have a name. It has 6 simple objects.

````julia
using TensorKit, MultiTensorKit, MPSKit, MPSKitModels
````

## Identifying the simple objects
We first need to select which fusion category we wish to use to grade the physical Hilbert space, and which fusion category to represent e.g. the symmetry category. Say we choose $\mathcal{D} = \mathsf{Vec A_4}$ for the physical Hilbert space and $\mathcal{C} = \mathcal{D}^*_{\mathcal{M}} = \mathsf{Rep_{A_4}}$ the symmetry category. This fixes the module category $\mathcal{M} = \mathsf{Vec}$. When referring to specific fusion and module categories, we will use this non-multifusion notation.

The easiest way to identify which elements of the multifusion category correspond to the subcategories we wish to use is ... (not sure yet how to do this yet)

Once we have identified the fusion and module categories, we now want to select the relevant objects we wish to place in our graded spaces. Unfortunately, due to the nature of how the N-symbol and F-symbol data are generated, the objects of the fusion subcategories are not ordered such that `label=1` corresponds to the unit object. Hence, the simplest way to find the unit object of a fusion subcategory is

````julia
one(A4Object(i,i,1))
````

Left and right units of subcategories are uniquely specified by their fusion rules. For example, the left unit of a subcategory $\mathcal{C}_{ij}$ is the simple object in $\mathcal{C}_i$ for which

$$ ^{}_a \mathbb{1} \times a = a \quad \forall a \in \mathcal{C}_i.$$

A similar condition uniquely defines the right unit of a subcategory. For fusion subcategories, a necessary condition is that the left and right units coincide.

Identifying the other simple objects of a (not necessarily fusion) category requires more work. We recommend a combination of the following to uniquely determine any simple object `a`:
- Check the dimension of the simple object: `dim(a)`
- Check the dual of the simple object: `dual(a)`
- Check how this simple object fuses with other (simple) objects: `Nsymbol(a,b,c)`

The dual object of some simple object $a$ of an arbitrary subcategory $\mathcal{C}_{ij}$ is defined as the unique object $a^* \in \mathcal{C}_{ji}$ satisfying

$$ ^{}_a \mathbb{1} \in a \times a^* \quad \text{and} \quad \mathbb{1}_a \in a^* \times a,$$

with multiplicity 1.
## Matrix product state simulations with MPSKit
TensorKit has been made compatible with the multifusion structure by keeping track of the relevant units in the fusion tree manipulations. With this, we can make `GradedSpace`s whose objects are in `A4Object`. With our example in mind, we first select the objects:

````julia
D0 = one(A4Object(2,2,1)) # unit object of VecA4
D1 = A4Object(2,2,2) # some self-dual object of VecA4

M = A4Object(1,2,1) # Vec

C0 = one(A4Object(1,1,1)) # unit object of RepA4
C1 = A4Object(1,1,1) # non-trivial 1d irrep
````

Afterwards, we build the physical and virtual space of the matrix product state:
````julia
P = Vect[A4Object](D0 => 1, D1 => 1)
V = Vect[A4Object](M => D) # D is the bond dimension
````
### Infinite case
Now, using MPKSit, we can perform matrix product state calculations. We construct some nearest-neighbour Hamiltonian and find the MPS representation of the ground state.
````julia
h = ones(ComplexF64, P ⊗ P ← P ⊗ P)
H = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(InfiniteChain(1)))
init = InfiniteMPS([P], [V])

gs, envs = find_groundstate(init, H, VUMPS())
````

Besides `VUMPS`, `IDMRG` and `IDMRG2` are as easy to run with the `A4Object` `Sector`.

A couple of MPSKit functions require an additional keyword argument `sector` (note the lowercase "s") specifying which sector to target within the function. These are:
- `transfer_spectrum`: the sector is selected by adding an auxiliary space to the *domain* of each eigenvector of the transfer matrix. Since in a full contraction the domain of the eigenvector lies in the opposite side of the physical space (labeled by objects in $\mathcal{D}$), the sectors lie in the symmetry category $\mathcal{C}$.
- `correlation_length`: since this function calls `transfer_spectrum`, the same logic applies.
- `excitations` with `QuasiparticleAnsatz`: similar to the previous functions, charged excitations are selected by adding a charged auxiliary space to the eigenvectors representing the quasiparticle states. 

### Finite case
There are minor differences to pay attention to when simulating matrix product states with a finite length. The first noticeable difference is in the `FiniteMPS` constructor itself to build an initial state. We must pass on a left and right virtual space to the keyword arguments `left` and `right`, since these would by default try to place a trivial space of the `Sector`, which does not exist for `BimoduleSector` due to the semisimple unit. Performing parallel calculations to the previous section now looks like

````julia
L = 10 # length of the MPS
init = FiniteMPS(L, P, V; left=V, right=V) # put Vec on the boundaries as well
H = @mpoham sum(h{i,j} for (i,j) in nearest_neighbours(FiniteChain(L)))

gs, envs = find_groundstate(init, H, DMRG())
````

`DMRG2` will run in a similar manner. Additional `sector` keywords are present for the following:
- `exact_diagonalization`: the `sector` keyword argument now requires an object in $\mathcal{D}$, since this is the fusion category which specifies the bond algebra from which the Hamiltonian is constructed. This is equivalent to adding a charged leg on the leftmost (or rightmost) virtual space of the MPS in conventional MPS cases.
- `excitations` with `QuasiparticleAnsatz`: see infinite case.