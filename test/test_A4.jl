using TensorKitSectors

testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKitSectors))), # TensorKitSectors root
    "test", "testsuite.jl"
)
include(testsuite_path)
using .SectorTestSuite

using MultiTensorKit
using Test, TestExtras
using Random

Random.seed!(1234)
rng = Random.Xoshiro(1234)

const MTK = MultiTensorKit

I = A4Object
Istr = TensorKitSectors.type_repr(I)
r = size(I)

println("----------------------")
println("|    Sector tests    |")
println("----------------------")
@testset "Sector test suite" verbose = true begin
    @time SectorTestSuite.test_sector(I)
end

@testset "$Istr construction and ordering" begin
    @test_throws DomainError I(1, 1, 0)
    @test_throws DomainError I(1, 1, -1)
    @test_throws DomainError I(0, 1, 1)
    @test_throws DomainError I(r + 1, 1, 1)
    @test_throws DomainError I(1, 1, MTK._numlabels(I, 1, 1) + 1)
    vals = collect(values(I))
    @test issorted(vals)
    @test length(vals) == length(values(I))
end

@testset "$Istr ($i, $j) basic properties" for i in 1:r, j in 1:r
    Cii_obs = I.(i, i, 1:MTK._numlabels(I, i, i))
    Cij_obs = I.(i, j, 1:MTK._numlabels(I, i, j))
    Cji_obs = I.(j, i, 1:MTK._numlabels(I, j, i))
    Cjj_obs = I.(j, j, 1:MTK._numlabels(I, j, j))
    c, m, mop, d = rand(rng, Cii_obs), rand(rng, Cij_obs), rand(rng, Cji_obs), rand(rng, Cjj_obs)

    if i == j
        @testset "Basic fusion properties" begin
            @test isunit(@testinferred(unit(c)))
            u = only(filter(isunit, Cii_obs))
            @test u == @testinferred(leftunit(u)) == @testinferred(rightunit(u)) ==
                @testinferred(unit(u))
        end
    else
        @testset "Basic module properties" begin
            @test isunit(m) == false
            @test isunit(mop) == false
            @test (isunit(@testinferred(leftunit(m))) && isunit(@testinferred(rightunit(m))))
            @test unit(c) == leftunit(m) == rightunit(mop)
            @test unit(d) == rightunit(m) == leftunit(mop)
            @test_throws DomainError unit(m)
            @test_throws DomainError unit(mop)
        end

        @testset "$Istr Fusion rules" begin
            argerr = ArgumentError("invalid fusion channel")
            # forbidden fusions
            for obs in [(c, d), (d, c), (m, m), (mop, mop), (d, m), (m, c), (mop, d), (c, mop)]
                @test isempty(⊗(obs...))
                @test_throws argerr Nsymbol(obs..., rand(rng, [c, m, mop, d]))
            end

            # allowed fusions
            for obs in [(c, c), (d, d), (m, mop), (mop, m), (c, m), (mop, c), (m, d), (d, mop)]
                @test !isempty(⊗(obs...))
            end

            @test Nsymbol(c, unit(c), c) == Nsymbol(d, unit(d), d) == 1

            @test_throws argerr Nsymbol(m, mop, d)
            @test_throws argerr Nsymbol(mop, m, c)
            @test_throws argerr Fsymbol(m, mop, m, mop, c, d)
        end
    end
end

@testset "$Istr units and duals of all objects" begin
    for a in values(I)
        ā = dual(a)
        @test dual(ā) == a
        @test Nsymbol(a, ā, leftunit(a)) == 1
        @test Nsymbol(ā, a, rightunit(a)) == 1
    end
end

println("-----------------------------")
println("|    F-symbol data tests    |")
println("-----------------------------")

# explicitly test everything related to F-symbols
# other option is to edit smallset to sample more
@testset "Unitarity of $Istr F-move" begin
    objs = collect(values(I))
    for a in objs, b in objs
        a.j == b.i || continue
        for c in objs
            b.j == c.i || continue
            @test SectorTestSuite.F_unitarity_test(a, b, c; atol = 1.0e-12, rtol = 1.0e-12)
        end
    end
end

@testset "Triangle equation" begin
    objects = collect(values(I))
    for a in objects, b in objects
        a.j == b.i || continue # skip if not compatible
        @test triangle_equation(a, b; atol = 1.0e-12, rtol = 1.0e-12)
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
                    c.j == d.i || continue # skip if not compatible
                    @test pentagon_equation(a, b, c, d; atol = 1.0e-12, rtol = 1.0e-12)
                end
            end
        end
    end
end
