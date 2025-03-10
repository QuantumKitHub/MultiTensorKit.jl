using MultiTensorKit
using TensorKitSectors
using Test, TestExtras

I = A4Object

@testset "Basic type properties" begin
    Istr = TensorKitSectors.type_repr(I)
    @test eval(Meta.parse(sprint(show, I))) == I
    @test eval(Meta.parse(TensorKitSectors.type_repr(I))) == I
end

@testset "Fusion Category $i" for i in 1:12
    objects = A4Object.(i, i, MultiTensorKit._get_dual_cache(I)[i][2])

    @testset "Basic properties" begin
        s = rand(objects, 3)
        @test eval(Meta.parse(sprint(show, s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test isone(@constinferred(one(s[1])))
        @constinferred dual(s[1])
        @constinferred dim(s[1]) # problem with this and 3 below
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

    @testset "Pentagon equation" begin
        for a in objects, b in objects, c in objects, d in objects
            @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
        end
    end
end

#TODO: add tests for module categories