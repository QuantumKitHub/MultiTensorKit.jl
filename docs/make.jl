using Documenter
using DocumenterCitations
using MultiTensorKit

pages = ["Home" => "index.md", "Library" => "lib/library.md",
         "Manual" => ["man/fusioncats.md", "man/multifusioncats.md"],
         "References" => "references.md"]


# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "MTKrefs.bib")
bib = CitationBibliography(bibpath; style=:authoryear)

makedocs(; sitename="MultiTensorKit Documentation", modules=[MultiTensorKit],
         authors="Boris De Vos, Laurens Lootens and Lukas Devos",
         pages=pages, pagesonly=true, plugins=[bib])

deploydocs(; repo="https://github.com/QuantumKitHub/MultiTensorKit.jl")