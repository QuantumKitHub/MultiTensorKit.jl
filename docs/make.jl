using Documenter
using MultiTensorKit

pages = ["Home" => "index.md", "Library" => "lib/library.md",
         "Manual" => ["man/fusioncats.md", "man/multifusioncats.md"],
         "References" => "references.md"]

makedocs(; sitename="MultiTensorKit Documentation", modules=[MultiTensorKit],
         authors="Boris De Vos, Laurens Lootens and Lukas Devos",
         pages=pages, pagesonly=true)

deploydocs(; repo="https://github.com/QuantumKitHub/MultiTensorKit.jl")