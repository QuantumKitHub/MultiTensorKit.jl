module TestSetup

export unitarity_test, rand_object, random_fusion, eval_show

using MultiTensorKit
using TensorKitSectors
using Random

const MTK = MultiTensorKit

Random.seed!(1234)

function unitarity_test(as::V, bs::V, cs::V) where {V <: AbstractVector{<:BimoduleSector}}
    @assert all(a.j == b.i for a in as, b in bs)
    @assert all(b.j == c.i for b in bs, c in cs)

    for a in as, b in bs, c in cs
        for d in ⊗(a, b, c)
            es = collect(intersect(⊗(a, b), map(dual, ⊗(c, dual(d)))))
            fs = collect(intersect(⊗(b, c), map(dual, ⊗(dual(d), a))))
            Fblocks = Vector{Any}()
            for e in es, f in fs
                Fs = Fsymbol(a, b, c, d, e, f)
                push!(Fblocks, reshape(Fs, (size(Fs, 1) * size(Fs, 2), size(Fs, 3) * size(Fs, 4))))
            end
            F = hvcat(length(fs), Fblocks...)
            isapprox(F' * F, one(F); atol = 1.0e-12, rtol = 1.0e-12) || return false
        end
    end
    return true
end

all_objects(::Type{<:BimoduleSector}, i::Int, j::Int) = [I(i, j, k) for k in 1:MTK._numlabels(I, i, j)]

function rand_object(I::Type{<:BimoduleSector}, i::Int, j::Int)
    obs = all_objects(I, i, j)
    ob = rand(obs)
    while isunit(ob) # unit of any fusion cat avoided
        ob = rand(obs)
    end

    return ob
end

function random_fusion(I::Type{<:BimoduleSector}, i::Int, j::Int, ::Val{N}) where {N} # for fusion tree tests
    N == 1 && return (rand_object(I, i, j),)
    tail = random_fusion(I, i, j, Val(N - 1))
    counter = 0

    Cs = all_objects(I, i, i)
    Ds = all_objects(I, j, j)
    Ms = all_objects(I, i, j)
    Mops = all_objects(I, j, i)
    allobs = vcat(Cs, Ds, Ms, Mops)
    s = rand(allobs)

    while isempty(⊗(s, first(tail))) && counter < 40
        counter += 1
        s = (counter < 40) ? rand(allobs) : leftunit(first(tail))
    end
    return (s, tail...)
end

"""
    eval_show(x)

Use `show` to generate a string representation of `x`, then parse and evaluate the resulting expression.
"""
function eval_show(x)
    str = sprint(show, x; context = (:module => @__MODULE__))
    ex = Meta.parse(str)
    return eval(ex)
end

end # end of module TestSetup
