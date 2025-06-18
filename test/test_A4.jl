using MultiTensorKit
using TensorKitSectors, TensorKit
using Test, TestExtras

I = A4Object

@testset "Basic type properties" verbose = true begin
    Istr = TensorKitSectors.type_repr(I)
    @test eval(Meta.parse(sprint(show, I))) == I
    @test eval(Meta.parse(TensorKitSectors.type_repr(I))) == I
end

@testset "Fusion Category $i" for i in 1:12
    objects = A4Object.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])

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

@testset "Pentagon equation" begin
    objects = collect(values(A4Object))
    for a in objects
        for b in objects
            a.j == b.i || continue # skip if not compatible
            for c in objects
                b.j == c.i || continue # skip if not compatible
                for d in objects
                    c.j == d.i || continue # skip if not compatible
                    @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
                end
            end
        end
    end
end

@testset "A4 Category ($i, $j) units and duals" for i in 1:12, j in 1:12
    Cij_obs = A4Object.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])

    s = rand(Cij_obs, 1)[1]
    @test eval(Meta.parse(sprint(show, s))) == s
    @test @constinferred(hash(s)) == hash(deepcopy(s))
    @test i == j ? isone(@constinferred(one(s))) :
          (isone(@constinferred(leftone(s))) && isone(@constinferred(rightone(s))))
    @constinferred dual(s)
    @test dual(s) == A4Object(j, i, MultiTensorKit._get_dual_cache(I)[2][i, j][s.label])
    @test dual(dual(s)) == s
end

@testset "A4 Category ($i, $j) left and right units" for i in 1:12, j in 1:12
    Cij_obs = A4Object.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])

    s = rand(Cij_obs, 1)[1]
    sp = Vect[A4Object](s => 1)
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