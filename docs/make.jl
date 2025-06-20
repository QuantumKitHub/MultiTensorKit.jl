using Documenter
using DocumenterCitations
using MultiTensorKit

pages = ["Home" => "index.md", "Library" => "lib/library.md",
         "Manual" => ["man/fusioncats.md", "man/multifusioncats.md"],
         "References" => "references.md"]

# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "MTKrefs.bib")
bib = CitationBibliography(bibpath; style=:authoryear)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/physics"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "tags" => "ams",
                                        "packages" => ["base", "ams", "autoload", "physics",
                                                       "mathtools"])))

makedocs(; sitename="MultiTensorKit.jl", modules=[MultiTensorKit],
         #  assets=["assets/custom.css"],
         authors="Boris De Vos, Laurens Lootens and Lukas Devos",
         pages=pages, pagesonly=true, plugins=[bib],
         format=Documenter.HTML(; mathengine=mathengine))

deploydocs(; repo="https://github.com/QuantumKitHub/MultiTensorKit.jl")