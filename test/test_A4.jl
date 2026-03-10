using TensorKitSectors

testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKitSectors))), # TensorKitSectors root
    "test", "testsuite.jl"
)
include(testsuite_path)
using .SectorTestSuite

using MultiTensorKit
using Test, TestExtras

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

@testset "$Istr ($i, $j) basic properties" for i in 1:r, j in 1:r
    Cii_obs = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
    Cij_obs = I.(i, j, MTK._get_dual_cache(I)[2][i, j])
    Cji_obs = I.(j, i, MTK._get_dual_cache(I)[2][j, i])
    Cjj_obs = I.(j, j, MTK._get_dual_cache(I)[2][j, j])
    c, m, mop, d = rand(Cii_obs), rand(Cij_obs), rand(Cji_obs), rand(Cjj_obs)

    if i == j
        @testset "Basic fusion properties" begin
            @test isunit(@testinferred(unit(c)))
            u = I.(i, i, MTK._get_dual_cache(I)[1][i])
            @test u == @testinferred(leftunit(u)) == @testinferred(rightunit(u)) ==
                @testinferred(unit(u))
            @test isunit(@testinferred(unit(c)))
            @test dual(c) == I.(i, i, MTK._get_dual_cache(I)[2][i, i][c.label])
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
            @test dual(m) == I.(j, i, MTK._get_dual_cache(I)[2][i, j][m.label])
        end

        @testset "$Istr Fusion rules" begin
            argerr = ArgumentError("invalid fusion channel")
            # forbidden fusions
            for obs in [(c, d), (d, c), (m, m), (mop, mop), (d, m), (m, c), (mop, d), (c, mop)]
                @test isempty(⊗(obs...))
                @test_throws argerr Nsymbol(obs..., rand([c, m, mop, d]))
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

println("-----------------------------")
println("|    F-symbol data tests    |")
println("-----------------------------")

# explicitly test everything related to F-symbols
# other option is to edit smallset to sample more
for i in 1:r, j in 1:r
    @testset "Unitarity of $Istr F-move ($i, $j)" begin
        if i == j
            @testset "Unitarity of fusion F-move ($i, $j)" begin
                fusion_objects = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
                for a in fusion_objects, b in fusion_objects, c in fusion_objects
                    @test SectorTestSuite.F_unitarity_test(a, b, c)
                end
            end
        end

        i != j || continue # do this part only when off-diagonal
        mod_objects = I.(i, j, MTK._get_dual_cache(I)[2][i, j])
        left_fusion_objects = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
        right_fusion_objects = I.(j, j, MTK._get_dual_cache(I)[2][j, j])

        # C x C x M -> M or D x D x Mop -> Mop
        @testset "Unitarity of left module F-move ($i, $j)" begin
            for a in left_fusion_objects, b in left_fusion_objects, A in mod_objects
                @test SectorTestSuite.F_unitarity_test(a, b, A)
            end
        end

        # M x D x D -> M or Mop x C x C -> Mop
        @testset "Unitarity of right module F-move ($i, $j)" begin
            for A in mod_objects, b in right_fusion_objects, c in right_fusion_objects
                @test SectorTestSuite.F_unitarity_test(A, b, c)
            end
        end

        # C x M x D -> M or D x Mop x C -> Mop
        @testset "Unitarity of bimodule F-move ($i, $j)" begin
            for a in left_fusion_objects, A in mod_objects, α in right_fusion_objects
                @test SectorTestSuite.F_unitarity_test(a, A, α)
            end
        end

        @testset "Unitarity of mixed module F-move ($i, $j) and opposite ($j, $i)" begin
            modop_objects = I.(j, i, MTK._get_dual_cache(I)[2][j, i])

            # C x M x Mop -> C or D x Mop x M -> D
            # M x Mop x C -> C or Mop x M x D -> D
            # Mop x C x M -> D or M x D x Mop -> C
            for a in left_fusion_objects, A in mod_objects, Aop in modop_objects
                @test SectorTestSuite.F_unitarity_test(a, A, Aop)
                @test SectorTestSuite.F_unitarity_test(A, Aop, a)
                @test SectorTestSuite.F_unitarity_test(Aop, a, A)
            end

            # M x Mop x M -> M or Mop x M x Mop -> Mop
            for A in mod_objects, Aop in modop_objects
                @test SectorTestSuite.F_unitarity_test(A, Aop, A)
            end
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

# include("TK_compat.jl")