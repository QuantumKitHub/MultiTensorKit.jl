using Documenter
using MultiTensorKit

makedocs(; sitename="MultiTensorKit Documentation", modules=[MultiTensorKit],
         authors="Boris De Vos, Laurens Lootens and Lukas Devos")

deploydocs(; repo="https://github.com/QuantumKitHub/MultiTensorKit.jl")