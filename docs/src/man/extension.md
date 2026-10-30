# [MultiTensorKit as an extension to TensorKit](@id extension)

This section will explain the internal changes to TensorKit which are required to extend the compatibility with fusion categories to multifusion ones.
Users who are unfamiliar with TensorKit are kindly guided towards the [TensorKit tutorial](https://jutho.github.io/TensorKit.jl/stable/man/tutorial/).

## As a `Sector`
MultiTensorKit is at its core an extension to [TensorKitSectors](https://github.com/QuantumKitHub/TensorKitSectors.jl), as it simply provides a new `Sector`, i.e. simple objects of a multifusion category, named `BimoduleSector`.

````julia
struct BimoduleSector{Name} <: Sector
    i::Int
    j::Int
    label::Int
end
````

`i` and `j` specify which subcategory $\mathcal{C}_{ij}$ we are considering, and `label` selects a particular simple object within that subcategory.
`Name` selects which multifusion category to work with, and is a `Symbol`.
As of now, only `:A4` is available, referring to the multifusion category consisting of $\mathsf{Rep(\mathbb{A}_4)}$ as the largest fusion category,  and all its Morita dual fusion categories and corresponding bimodule categories.
The fusion of these `BimoduleSector`s is then defined by the fusion rules of the multifusion category.
In particular,
````julia
a = BimoduleSector{:A4}(i, j, label1)
b = BimoduleSector{:A4}(k, l, label2)
a ⊗ b # empty unless j == k
````

The fusion rules are read in via the artifact labeled "fusiondata", and are extracted at runtime when calling the fusion of two `BimoduleSector`s (or `Nsymbol`) for the first time.
These data are cached in a hash table for later use.

The graphical calculus of multifusion categories naturally uses colorings to distinguish between the different fusion categories and bimodule categories.
The notion of a *left* and *right* unit of some (simple) object is introduced not only as the simple object(s) which fuse trivially with the considered object, but also to uniquely identify these colorings.
These are called with the functions `TensorKitSectors.leftunit` and `TensorKitSectors.rightunit`.
For the usual case of just one fusion category, these both coincide with the unique unit object.
The `TensorKitSectors.UnitStyle` trait is introduced to differentiate between fusion categories and multifusion categories, returning `SimpleUnit()` and `GenericUnit()`, respectively.
This trait is used to specialise certain functions within TensorKit (see below).
`TensorKitSectors.allunits` is also introduced to return all the simple unit objects of the (multi)fusion category.
This function must be defined for every multifusion category to be compatible with TensorKit.

Via the fusion rules, the left and right units of all `BimoduleSector`s along with their duals are extracted and cached.
The imposed fusion properties for the `BimoduleSector` type is the most general one that appears in the fusion between any two `BimoduleSector`s.
For example, if one of the diagonal fusion categories has fusion rules with multiplicities, then the entire `BimoduleSector` type must be set to have `TensorKitSectors.FusionStyle(::Type{<:BimoduleSector}) = GenericFusion()`.
This is the case for the $\mathbb{A}_4$ `BimoduleSector`, since $\mathsf{Rep(\mathbb{A}_4)}$ is one of the diagonal fusion categories and has fusion rules with multiplicities.
Besides the fusion rules, the F-symbols are also read in from the artifact, and are stored in a hash table for later use.

It is currently not possible to define a non-trivial braiding on the `BimoduleSector`s, so these are set to be non-braided: `TensorKitSectors.BraidingStyle(::Type{<:BimoduleSector}) = NoBraiding()`.
This is especially important when working with matrix product states, as all algorithms are required to remain planar since no (half-)braiding is available to perform crossings of fusion trees.

## As a symmetry in TensorKit
Since `BimoduleSector`s are `Sector`s, they can be used as symmetries in TensorKit.
This way, we can construct symmetric tensors with the symmetries of the multifusion category, which are more general than those of the fusion categories.
In particular, the vector spaces graded by these `BimoduleSector`s may contain simple objects of not only the fusion subcategories, but also the bimodule categories.

Certain changes within TensorKit were required to make it compatible with the multifusion categorical structure, largely due to the fact that "the" unit object is semi-simple.
It is thus important that the correct unit object is considered when manipulating fusion trees; this is achieved through identifying colorings and the use of `leftunit` and `rightunit`.
Every fusion tree manipulation which previously involved "the" unit object, now involves the `leftunit` and `rightunit` of some neighboring sector.
An important example of this is explained in the previous section on [Opposite module categories](@ref opposite_module_categories), namely the mapping of a splitting vertex to a fusion vertex via a B-move.

Just like how colorings at the level of the simple objects (`BimoduleSector`s) check whether certain fusions are allowed, vector spaces graded by these simple objects also have to be checked for compatibility, since these essentially represent in general the direct sum of simple objects.
`TensorKit.leftunitspace` and `TensorKit.rightunitspace` are the `GradedSpace` equivalents to `leftunit` and `rightunit`, which check whether the coloring of the composite object grading the space is uniform, after which they return the one-dimensional space with the corresponding `leftunit` or `rightunit`.
Clearly, `leftunitspace` and `rightunitspace` coincide for fusion categories; this defaults to `TensorKit.unitspace`.
TensorKit does not allow to construct the `unitspace` of a multifusion category directly with this function, since this would construct a `GradedSpace` with ambiguous coloring, which is currently not supported by TensorKit and downstream packages.
`leftunitspace` and `rightunitspace` are used within the functions `TensorKit.insertleftunit` and `TensorKit.insertrightunit`, which also check coloring compatibility.
Similarly, `TensorKit.removeunit` is used to remove unit spaces, and `TensorKit.isunitspace` checks this property.

## MultiTensorKit compatibility with MPSKit

The main change within MPSKit is very similar to the fusion tree manipulations in TensorKit, namely the use of `leftunit` and `rightunit` to identify the correct unit object.
In the case of MPSKit, trivial spaces are used everywhere, from the boundary of a finite matrix product state to the virtual spaces of a Hamiltonian written as a matrix product operator.
Since no (half-)braiding is available for the `BimoduleSector`s, all algorithms are made planar, and thus all braiding tensors are removed.
Previous braidings involving unit spaces are replaced by a termination and reintroduction of the legs without crossing, which is achieved by the use of `removeunit`, `insertleftunit` and `insertrightunit`.

At the level of the MPS, the correct unit object can be identified through the use of `leftunit(ψ::AbstractMPS)` and `rightunit(ψ::AbstractMPS)`.
When the virtual space of the MPS is graded by a diagonal `BimoduleSector`, i.ea unitary fusion category, then these all coincide with the unique unit object of that fusion category.
However, when the virtual space is graded by an off-diagonal `BimoduleSector`, i.e. a bimodule category, then the left and right units are different.
In the canonical fusion tree basis of the MPS, the right coloring is that on the side with the physical legs, such that Hamiltonians and other physical operators always have the same coloring as the right unit of the MPS.
Similarly, excitations of an MPS are labeled by `BimoduleSector`s with the same coloring as the left unit of the MPS, since the auxiliary charge leg of the excitation is attached on the other side of the MPS to the virtual level.
This is illustrated in the following figure.

```@raw html
<img src="../img/anyonchain_excitation.svg" alt="" width="70%"/>
```

Here, $\mathcal{D}$ is the fusion category grading the physical space of the MPS, $\mathcal{R}$ is the right $\mathcal{D}$-module category grading the virtual space of the MPS.
$\mathcal{E}$ is the Morita dual fusion category of $\mathcal{D}$ with respect to $\mathcal{R}$, which is the category labelling the excitations of the MPS.
In this case, the left unit of the MPS is in $\mathcal{E}$, and the right unit in $\mathcal{D}$.