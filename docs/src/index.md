# MultiTensorKit

**TensorKit extension to multifusion categories**

## Package summary
MultiTensorKit.jl provides the user a package to work with multifusion categories, the extension of regular fusion categories where the unit is no longer simple and unique.
Multifusion categories naturally embed the structure of module categories over fusion categories. Hence, MultiTensorKit.jl allows not only the fusion of objects within the same
fusion category (as TensorKit.jl), but also the fusion with and between module categories over these fusion categories. 

MultiTensorKit.jl is built to be compatible with TensorKit, thus allowing the construction of symmetric tensors with new symmetries due to the module structure. Through this,
tensor network simulations of quantum many-body systems with aid of [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) can be performed.

## Table of contents

```@contents
Pages = ["man/fusioncats.md", "man/multifusioncats.md","lib/library.md", "references.md"]
Depth = 2
```

## Installation

MultiTensorKit.jl is currently not registered to the Julia General Registry. You can install the package as
```
pkg> add https://github.com/QuantumKitHub/MultiTensorKit.jl.git
```

## Usage

As the name suggests, MultiTensorKit is an extension of [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) and
[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl). Therefore, we recommend including TensorKit 
to your project. Additionally, MultiTensorKit was made to be functional with [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)
and [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) for Matrix Product State (MPS) calculations, supporting symmetries
which go beyond TensorKit.

