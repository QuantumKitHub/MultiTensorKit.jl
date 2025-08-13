using MultiTensorKit
using TensorKitSectors
using Random

const MTK = MultiTensorKit

Random.seed!(1234)

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

all_objects(::Type{<:BimoduleSector}, i::Int, j::Int) = [I(i, j, k) for k in 1:MTK._numlabels(I, i, j)]

function rand_object(I::Type{<:BimoduleSector}, i::Int, j::Int)
    obs = all_objects(I, i, j)
    ob = rand(obs)
    if i == j
        while ob == one(ob) # unit of any fusion cat avoided
            ob = rand(obs)
        end
    end
    return ob
end

function random_fusion(I::Type{<:BimoduleSector}, i::Int, j::Int, N::Int) # for fusion tree tests
    Cs = all_objects(I, i, i)
    Ds = all_objects(I, j, j)
    Ms = all_objects(I, i, j)
    Mops = all_objects(I, j, i)
    allobs = vcat(Cs, Ds, Ms, Mops)

    in = nothing
    out = nothing
    while in === nothing
        out = ntuple(n -> rand(allobs), N)
        try
            in = rand(collect(⊗(out...)))
        catch e
            if isa(e, AssertionError)
                in = nothing
            else
                rethrow(e)
            end 
        end
    end
    return out
end

# for fusion tree merge test
function safe_tensor_product(x::I, y::I) where {I<:BimoduleSector}
    try
        return x ⊗ y
    catch e
        if e isa AssertionError
            return nothing
        else
            rethrow(e)
        end
    end
end