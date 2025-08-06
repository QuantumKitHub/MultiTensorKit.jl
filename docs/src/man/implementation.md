explain the f-symbol and n-symbol storage system

# MultiTensorKit implementation: $\mathsf{Rep(A_4)}$ as a guiding example
This tutorial is dedicated to explaining how MultiTensorKit was implemented to be compatible with with TensorKit and MPSKit for matrix product state simulations. In particular, we will be making a generalised anyonic spin chain. We will demonstrate how to reproduce the entanglement spectra found in [Lootens_2024](@cite). The model considered there is a spin-1 Heisenberg model with additional terms to break the usual $\mathsf{U_1}$ symmetry to $\mathsf{Rep(A_4)}$, while having a non-trivial phase diagram and relatively easy Hamiltonian to write down.

This will be done with the `A4Object = BimoduleSector{A4}` `Sector`, which is the multifusion category which contains the structure of the module categories over $\mathsf{Rep(A_4)}$. Since there are 7 module categories, `A4Object` is a $r=7$ multifusion category. There are 3 fusion categories up to equivalence:
- ``\mathsf{Vec_{A_4}}``: the category of $\mathsf{A_4}$-graded vector spaces. The group $\mathsf{A}_4$ is order $4!/2 = 12$. It has thus 12 objects.
- ``\mathsf{Rep(A_4)}``: the irreducible representations of the group $\mathsf{A}_4$, of which there are 4. One is the trivial representation, two are one-dimensional non-trivial and the last is three-dimensional.
- ``\mathsf{Rep(H)}``: the representation category of some Hopf algebra which does not have a name. It has 6 simple objects.

For this example, we will require the following packages:
````julia
using TensorKit, MultiTensorKit, MPSKit, MPSKitModels, Plots
````

## Identifying the simple objects
We first need to select which fusion category we wish to use to grade the physical Hilbert space, and which fusion category to represent e.g. the symmetry category. In our case, we are interested in selecting $\mathcal{D} = \mathsf{Rep(A_4)}$ for the physical Hilbert space. We know the module categories over $\mathsf{Rep(G)}$ to be $\mathsf{Rep^\psi(H)}$ for a subgroup $\mathsf{H}$ and 2-cocycle $\psi$. Thus, the 7 module categories $\mathcal{M}$ one can choose over $\mathsf{Rep(A_4)}$ are

- ``\mathsf{Rep(A_4)}`` itself as the regular module category,
- ``\mathsf{Vec}``: the category of vector spaces,
- ``\mathsf{Rep(\mathbb{Z}_2)}``,
- ``\mathsf{Rep(\mathbb{Z}_3)}``,
- ``\mathsf{Rep(\mathbb{Z}_2 \times \mathbb{Z}_2)}``,
- ``\mathsf{Rep^\psi(\mathbb{Z}_2 \times \mathbb{Z}_2)}``,
- ``\mathsf{Rep^\psi(A_4)}``.
  
When referring to specific fusion and module categories, we will use this non-multifusion notation.

The easiest way to identify which elements of the multifusion category correspond to the subcategories we wish to use is ... (not sure yet how to do this yet)

Now that we have identified the fusion and module categories, we want to select the relevant objects we wish to place in our graded spaces. Unfortunately, due to the nature of how the N-symbol and F-symbol data are generated, the objects of the fusion subcategories are not ordered such that `label=1` corresponds to the unit object. Hence, the simplest way to find the unit object of a fusion subcategory is

````julia
one(A4Object(i,i,1))
````

Left and right units of subcategories are uniquely specified by their fusion rules. For example, the left unit of a subcategory $\mathcal{C}_{ij}$ is the simple object in $\mathcal{C}_i$ for which

$$^{}_a \mathbb{1} \times a = a \quad \forall a \in \mathcal{C}_i.$$

A similar condition uniquely defines the right unit of a subcategory. For fusion subcategories, a necessary condition is that the left and right units coincide.

Identifying the other simple objects of a (not necessarily fusion) category requires more work. We recommend a combination of the following to uniquely determine any simple object `a`:
- Check the dimension of the simple object: `dim(a)`
- Check the dual of the simple object: `dual(a)`
- Check how this simple object fuses with other (simple) objects: `Nsymbol(a,b,c)`

The dual object of some simple object $a$ of an arbitrary subcategory $\mathcal{C}_{ij}$ is defined as the unique object $a^* \in \mathcal{C}_{ji}$ satisfying

$$^{}_a \mathbb{1} \in a \times a^* \quad \text{and} \quad \mathbb{1}_a \in a^* \times a,$$

with multiplicity 1.
## Constructing the Hamiltonian and matrix product state
TensorKit has been made compatible with the multifusion structure by keeping track of the relevant units in the fusion tree manipulations. With this, we can make `GradedSpace`s whose objects are in `A4Object`: 

````julia
D1 = A4Object(6, 6, 1) # unit in this case
D2 = A4Object(6, 6, 2) # non-trivial 1d irrep
D3 = A4Object(6, 6, 3) # non-trivial 1d irrep
D4 = A4Object(6, 6, 4) # 3d irrep
````
Since we want to replicate a spin-1 Heisenberg model, it makes sense to use the 3-dimensional irrep to grade the physical space, and thus construct our Hamiltonian. We don't illustrate here how to derive the considered Hamiltonian in a $\mathsf{Rep(A_4)}$ basis, but simply give it. For now, we construct a finite-size spin chain; below we will repeat the calculation for an infinite system.

````julia
P = Vect[A4Object](D4 => 1) # physical space
T = ComplexF64
# usual Heisenberg part
h1_L = TensorMap(zeros, T, P ⊗ P ← P)
h1_R = TensorMap(zeros, T, P ← P ⊗ P)
block(h1_L, D4) .= [0; 1]
block(h1_R, D4) .= [0 1;]
@plansor h1[-1 -2; -3 -4] := h1_L[-1 1; -3] * h1_R[-2; 1 -4]

# biquadratic term
h2_L = TensorMap(zeros, T, P ⊗ Vect[A4Object](D1 => 1, D2 => 1, D3 => 1) ← P)
h2_R = TensorMap(ones, T, P ← Vect[A4Object](D1 => 1, D2 => 1, D3 => 1) ⊗ P)
block(h2_L, D4) .= [4 / 3; 1 / 3; 1 / 3]
@plansor h2[-1 -2; -3 -4] := h2_L[-1 1; -3] * h2_R[-2; 1 -4]

# anti-commutation term
h3_L = TensorMap(zeros, T, P ⊗ P ← P)
h3_R = TensorMap(zeros, T, P ← P ⊗ P)
block(h3_L, D4) .= [1; 0]
block(h3_R, D4) .= [0 1;]
@plansor h3[-1 -2; -3 -4] := h3_L[-1 1; -3] * h3_R[-2; 1 -4]

L = 60
J1 = -2.0 # probing the A4 SSB phase first
J2 = -5.0
lattice = FiniteChain(L)
H1 = @mpoham sum(-2 * h1{i,j} for (i, j) in nearest_neighbours(lattice))
H2 = @mpoham sum(h2{i,j} for (i, j) in nearest_neighbours(lattice))
H3 = @mpoham sum(2im * h3{i,j} for (i, j) in nearest_neighbours(lattice))

H = H1 + J1 * H2 + J3 * H3
````

For the matrix product state, we will select $\mathsf{Vec}$ as the module category for now:
````julia
M = A4Object(1, 6, 1) # Vec
````
and construct the finite MPS:
````julia
D = 40 # bond dimension
V = Vect[A4Object](M => D)
Vb = Vect[A4Object](M => 1) # non-degenerate boundary virtual space
init_mps = FiniteMPS(L, P, V; left=Vb, right=Vb)
````

!!! warning "Important"
    We must pass on a left and right virtual space to the keyword arguments `left` and `right` of the `FiniteMPS` constructor, since these would by default try to place a trivial space of the `Sector`, which does not exist for any `BimoduleSector` due to the semisimple unit. 

## DMRG2 and the entanglement spectrum
We can now look to find the ground state of the Hamiltonian with two-site DMRG. We use this instead of the "usual" one-site DMRG because the two-site one will smartly fill up the blocks of the local tensor during the sweep, allowing one to initialise as a product state in one block and more likely avoid local minima, a common occurence in symmetric tensor network simulations. 
````julia
dmrg2alg = DMRG2(;verbosity=2, tol=1e-7, trscheme=truncbelow(1e-4))
ψ, _ = find_groundstate(init_mps, H, dmrg2alg)
````
The truncation scheme keyword argument is mandatory when calling `DMRG2` in MPSKit. Here, we choose to truncate such that all singular values are larger than $10^{-4}$, while setting the default tolerance for convergence to $10^{-7}$. More information on this can be found in the [MPSKit](https://github.com/QuantumKitHub/MPSKit.jl) documentation. To run one-site DMRG anyway, use `DMRG` which does not require a truncation scheme.

Now that we've found the ground state, we can compute the entanglement spectrum in the middle of the chain.
````julia
spec = entanglement_spectrum(ψ, round(Int, L/2))
````
This returns a dictionary which maps the objects grading the virtual space to the singular values. In this case, there is one key corresponding to $\mathsf{Vec}$. We can also immediately return a plot of this data by the following:
````julia
entanglementplot(ψ;site=round(Int, L/2))
````
This plot will show the singular values per object, as well as include the "effective" bond dimension, which is simply the dimension of the virtual space where we cut the system. #TODO: actually include the plot 

## Search for the correct dual model

Consider a quantum lattice model with its symmetries determing the phase diagram. For every phase in the phase diagram, the dual model for which the ground state maximally breaks all symmetries spontaneously is the one where the entanglement is minimised and the tensor network is represented most efficiently [Lootens_2024](@cite). Let us confirm this result, starting with the $\mathsf{Rep(A_4)}$ spontaneous symmetry breaking phase. The code will look exactly the same as above, except the virtual space of the MPS will change to be graded by the other module categories:

````julia
module_numlabels(i) = MultiTensorKit._numlabels(A4Object, i, 6) 
V = Vect[A4Object](A4Object(i, 6, label) => D for label in 1:module_numlabels(i))
Vb = Vect[A4Object](c => 1 for c in first(sectors(V))) # not all charges on boundary, play around with what is there
````

#TODO: show all the plots

!!! note "Additional functions and keyword arguments"
    Certain commonly used functions within MPSKit require extra keyword arguments to be compatible with multifusion MPS simulations. In particular, the keyword argument `sector` (note the lowercase "s") appears in 
    - `transfer_spectrum`: the sector is selected by adding an auxiliary space to the *domain* of each eigenvector of the transfer matrix. Since in a full contraction the domain of the eigenvector lies in the opposite side of the physical space (labeled by objects in $\mathcal{D} = \mathsf{Rep(A_4)}$), the sectors lie in the symmetry category $\mathcal{C} = \mathcal{D^*_M}$.
    - `correlation_length`: since this function calls `transfer_spectrum`, the same logic applies.
    - `excitations` with `QuasiparticleAnsatz`: similar to the previous functions, charged excitations are selected by adding a charged auxiliary space to the eigenvectors representing the quasiparticle states. 
    - `exact_diagonalization`: the `sector` keyword argument now requires an object in $\mathcal{D}$, since this is the fusion category which specifies the bond algebra from which the Hamiltonian is constructed. This is equivalent to adding a charged leg on the leftmost (or rightmost) virtual space of the MPS in conventional MPS cases.

## Differences with the infinite case
We can repeat the above calcalations also for an infinite system. The `lattice` variable will change, as well as the MPS constructor and the algorithm:
````julia
lattice = InfiniteChain(1)
init = InfiniteMPS([P], [V])
inf_alg = VUMPS(; verbosity=2, tol=1e-7)
````

Besides `VUMPS`, `IDMRG` and `IDMRG2` are as easy to run with the `A4Object` `Sector`. It is also clear that boundary terms do not play a role in this case.