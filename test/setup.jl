using MultiTensorKit
using TensorKitSectors, TensorKit
using Test, TestExtras

const MTK = MultiTensorKit

function unitarity_test(as::Vector{I}, bs::Vector{I},
                        cs::Vector{I}) where {I<:BimoduleSector}
    @assert all(a.j == b.i for a in as, b in bs)
    @assert all(b.j == c.i for b in bs, c in cs)

    for a in as, b in bs, c in cs
        for d in ⊗(a, b, c)
            es = collect(intersect(⊗(a, b), map(dual, ⊗(c, dual(d)))))
            fs = collect(intersect(⊗(b, c), map(dual, ⊗(dual(d), a))))
            Fblocks = Vector{Any}()
            for e in es
                for f in fs
                    Fs = Fsymbol(a, b, c, d, e, f)
                    push!(Fblocks,
                          reshape(Fs,
                                  (size(Fs, 1) * size(Fs, 2),
                                   size(Fs, 3) * size(Fs, 4))))
                end
            end
            F = hvcat(length(fs), Fblocks...)
            isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12) || return false
        end
    end
    return true
end
