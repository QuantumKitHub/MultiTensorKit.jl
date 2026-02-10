# MultiTensorKit as an extension to TensorKit

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
As of now, only `:A4` is available, referring to the multifusion category consisting of $\mathsf{Rep(A_4)}$ as the largest fusion category,  and all its Morita dual fusion categories and corresponding bimodule categories.
The fusion of these `BimoduleSector`s is then defined by the fusion rules of the multifusion category.
In particular,
````julia
a = BimoduleSector{:A4}(i, j, label1)
b = BimoduleSector{:A4}(k, l, label2)
a ⊗ b # empty unless j == k
````

The fusion rules are read in via the artifact labeled "fusiondata", and are extracted at runtime when calling the fusion of two `BimoduleSector`s (or `Nsymbol`) for the first time.
These data are cached in a hash table for later use.

A consequence of the multifusion structure is the colorings used in the graphical calculus of fusions.
A natural introduction is the notion of a *left* and *right* unit of some (simple) object in the multifusion category.
Clearly, for the usual case of just one fusion category, these both coincide with the unique unit object.
Via the fusion rules, the left and right units of all `BimoduleSector`s along with their duals are also extracted and cached.
Since the most general fusion rule possible is one with multiplicities involved, the entire `BimoduleSector` is set to have `TensorKitSectors.FusionStyle(::Type{<:BimoduleSector}) = GenericFusion()`.

Going beyond the ring structure, the F-symbols are also read in from the artifact, and are stored in a hash table for later use.
The F-symbols are then used to perform F-moves on `BimoduleSector`s, which is required to perform recouplings of fusion trees when doing e.g. contractions of tensors with these categories grading the vector spaces.

Due to the nature of the multifusion category, it is currently not possible to define a non-trivial braiding on the `BimoduleSector`s, so these are set to be non-braided: `TensorKitSectors.BraidingStyle(::Type{<:BimoduleSector}) = NoBraiding()`.
This is especially important when working with matrix product states, as all algorithms are required to remain planar, since no (half-)braiding is available to perform crossings of fusion trees.

## As a symmetry in TensorKit
Since `BimoduleSector`s are `Sector`s, they can be used as symmetries in TensorKit.
This way, we can construct symmetric tensors with the symmetries of the multifusion category, which are more general than those of the fusion categories.
In particular, the vector spaces graded by these `BimoduleSector`s are not only graded by the simple objects of the fusion categories, but also by the simple objects of the bimodule categories.
This allows for more general tensor network simulations of quantum many-body systems with symmetries which go beyond those of fusion categories.

Certain changes within TensorKit were required to make it compatible with the multifusion categorical structure.
In particular, the presence of a simple unit object for every fusion category on the diagonal of the multifusion category, along with the off-diagonal nature of the simple objects of the bimodule categories, required some internal changes to the way unit objects were treated in TensorKit.
Most notably, the unit object is no longer unique, and thus it is of utmost importance that the correct unit object is considered when performing tensor contractions at the level of the fusion trees.
This is achieved precisely through colorings and the use of `leftunit` and `rightunit`.
For this reason, every fusion tree manipulation which previously involved "the" unit object, now involves the `leftunit` and `rightunit` of some neighboring sector in the manipulation to identify the correct color.
An important example of this is explained in the previous section [#TODO: add opposite module categories reference], namely the mapping of a splitting vertex to a fusion vertex through the B-move.

# MultiTensorKit compatibility with MPSKit

This section will briefly explain the changes within MPSKit which are required to make it compatible with MultiTensorKit.
For a more practical explanation, users are kindly guided towards the next section [#TODO: add implementation reference].

The main change within MPSKit is very similar to the fusion tree manipulations in TensorKit, namely the use of `leftunit` and `rightunit` to identify the correct unit object.
In the case of MPSKit, trivial spaces are used everywhere, from the boundary of a finite MPS to the virtual spaces of a Hamiltonian written in MPO form.
Additionally, multiple tensor contractions made use of braiding tensors to perform crossings of legs of the MPS/MPOs.
Since no (half-)braiding is available for the `BimoduleSector`s, all algorithms had to be made planar, and thus all braiding tensors were removed.
Since all braidings were trivial, this was dealt with by simply removing the braiding tensors and replacing the crossing of legs with a termination and reintroduction of the legs without crossing.
This is achieved through `TensorKit.removeunit` and `TensorKit.insertleftunit`/`TensorKit.insertrightunit`, which remove and insert spaces with the correct unit object, respectively, based on the grading of neighboring vector spaces.
At the level of the MPS, the correct unit object can be identified through the use of `leftunit` and `rightunit`.
When the virtual space of the MPS is graded by a diagonal `BimoduleSector`, i.ea unitary fusion category, then these all coincide with the unique unit object of that fusion category.
However, when the virtual space is graded by an off-diagonal `BimoduleSector`, i.e. a bimodule category, then the left and right units are different, and thus it is important to identify the correct one when performing MPS algorithms.
For example, Hamiltonians should always have the same coloring as the right unit of the MPS, since they are contracted at the physical level of the MPS.
Similarly, excitations of an MPS are labeled by `BimoduleSector`s with the same coloring as the left unit of the MPS, since the auxiliary charge leg of the excitation is attached on the other side of the MPS to the virtual level.