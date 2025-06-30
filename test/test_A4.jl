using MultiTensorKit
using TensorKitSectors, TensorKit
using Test, TestExtras

I = A4Object
Istr = TensorKitSectors.type_repr(I)
r = size(I)

@testset "$Istr Basic type properties" verbose = true begin
    @test eval(Meta.parse(sprint(show, I))) == I
    @test eval(Meta.parse(TensorKitSectors.type_repr(I))) == I
end

@testset "$Istr Fusion Category $i" for i in 1:r
    objects = I.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])

    @testset "Basic properties" begin
        s = rand(objects, 3)
        @test eval(Meta.parse(sprint(show, s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test isone(@constinferred(one(s[1])))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobeniusschur(s[1])
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
    end

    @testset "Unitarity of F-move" begin
        for a in objects, b in objects, c in objects
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
                @test isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12)
            end
        end
    end
end

for i in 1:r, j in 1:r # M x D (or Mop x C)
    i != j || continue # skip if fusion category
    @testset "$Istr right module category $i, $j" begin
        mod_objects = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])
        fusion_objects = I.(j, j, MultiTensorKit._get_dual_cache(I)[2][j, j])

        @testset "Unitarity of module F-move $i, $j" begin
            for A in mod_objects, α in fusion_objects, β in fusion_objects
                (A.j == α.i && α.j == β.i) || continue # skip if not compatible
                for B in ⊗(A, α, β)
                    Cs = collect(intersect(⊗(A, α), map(dual, ⊗(β, dual(B)))))
                    γs = collect(intersect(⊗(α, β), map(dual, ⊗(dual(B), A))))
                    Fblocks = Vector{Any}()
                    for C in Cs
                        for γ in γs
                            Fs = Fsymbol(A, α, β, B, C, γ)
                            push!(Fblocks,
                                  reshape(Fs,
                                          (size(Fs, 1) * size(Fs, 2),
                                           size(Fs, 3) * size(Fs, 4))))
                        end
                    end
                    F = hvcat(length(γs), Fblocks...)
                    @test isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12)
                end
            end
        end
    end
end

for i in 1:7, j in 1:7 # C x M (or D x Mop)
    i != j || continue # skip if fusion category
    @testset "$Istr left module category $i, $j unitarity check" begin
        mod_objects = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])
        fusion_objects = I.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])

        @testset "Unitarity of left module F-move" begin
            for a in fusion_objects, b in fusion_objects, A in mod_objects  # written for M as left C-module category
                (a.j == b.i && b.j == A.i) || continue # skip if not compatible
                for B in ⊗(a, b, A)
                    cs = collect(intersect(⊗(a, b), map(dual, ⊗(A, dual(B))))) # equivalent of es
                    Cs = collect(intersect(⊗(b, A), map(dual, ⊗(dual(B), a)))) # equivalent of fs
                    Fblocks = Vector{Any}()
                    for c in cs
                        for C in Cs
                            Fs = Fsymbol(a, b, A, B, c, C)
                            push!(Fblocks,
                                  reshape(Fs,
                                          (size(Fs, 1) * size(Fs, 2),
                                           size(Fs, 3) * size(Fs, 4))))
                        end
                    end
                    F = hvcat(length(Cs), Fblocks...)
                    isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12) ||
                        @show A, a, b, B, C, c, F
                end
            end
        end
    end
end

for i in 1:7, j in 1:7 # bimodule check unitarity
    i != j || continue # skip if fusion category
    @testset "$Istr bimodule category $i, $j" begin
        C_objects = I.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])
        mod_objects = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])
        D_objects = I.(j, j, MultiTensorKit._get_dual_cache(I)[2][j, j])

        @testset "Unitarity of bimodule F-move" begin
            for a in C_objects, A in mod_objects, α in D_objects
                (a.j == A.i && A.j == α.i) || continue # skip if not compatible
                for B in ⊗(a, A, α)
                    Cs = collect(intersect(⊗(a, A), map(dual, ⊗(α, dual(B))))) # equivalent of es
                    Ds = collect(intersect(⊗(A, α), map(dual, ⊗(dual(B), a)))) # equivalent of fs
                    Fblocks = Vector{Any}()
                    for C in Cs
                        for D in Ds
                            Fs = Fsymbol(a, A, α, B, C, D)
                            push!(Fblocks,
                                  reshape(Fs,
                                          (size(Fs, 1) * size(Fs, 2),
                                           size(Fs, 3) * size(Fs, 4))))
                        end
                    end
                    count += 1
                    F = hvcat(length(Ds), Fblocks...)
                    isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12) ||
                        @show A, a, α, B, C, D, F
                end
            end
        end
    end
end

@testset "$Istr Pentagon equation" begin
    objects = collect(values(I))
    for a in objects
        for b in objects
            a.j == b.i || continue # skip if not compatible
            for c in objects
                b.j == c.i || continue # skip if not compatible
                for d in objects
                    c.j == d.i || continue # skip if not compatible #TODO: check if this is right
                    @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
                end
            end
        end
    end
end

@testset "$Istr ($i, $j) units and duals" for i in 1:r, j in 1:r
    Cij_obs = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])

    s = rand(Cij_obs)
    @test eval(Meta.parse(sprint(show, s))) == s
    @test @constinferred(hash(s)) == hash(deepcopy(s))
    @test i == j ? isone(@constinferred(one(s))) :
          (isone(@constinferred(leftone(s))) && isone(@constinferred(rightone(s))))
    @constinferred dual(s)
    @test dual(s) == I.(j, i, MultiTensorKit._get_dual_cache(I)[2][i, j][s.label])
    @test dual(dual(s)) == s
end

@testset "$Istr ($i, $j) left and right units" for i in 1:r, j in 1:r
    Cij_obs = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])

    s = rand(Cij_obs, 1)[1]
    sp = Vect[I](s => 1)
    W = sp ← sp
    for T in (Float32, ComplexF64)
        t = @constinferred rand(T, W)

        for a in 1:2
            tl = @constinferred insertleftunit(t, Val(a))
            @test numind(tl) == numind(t) + 1
            @test space(tl) == insertleftunit(space(t), a)
            @test scalartype(tl) === T
            @test t.data === tl.data
            @test @constinferred(removeunit(tl, $(a))) == t

            tr = @constinferred insertrightunit(t, Val(a))
            @test numind(tr) == numind(t) + 1
            @test space(tr) == insertrightunit(space(t), a)
            @test scalartype(tr) === T
            @test t.data === tr.data
            @test @constinferred(removeunit(tr, $(a + 1))) == t
        end

        @test_throws ErrorException insertleftunit(t) # default should error here
        @test insertrightunit(t) isa TensorMap
        @test_throws ErrorException insertleftunit(t, numind(t) + 1) # same as default
        @test_throws ErrorException insertrightunit(t, numind(t) + 1) # not same as default

        t2 = @constinferred insertrightunit(t; copy=true)
        @test t.data !== t2.data
        for (c, b) in blocks(t)
            @test b == block(t2, c)
        end
        @test @constinferred(removeunit(t2, $(numind(t2)))) == t
    end
end