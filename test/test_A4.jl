using MultiTensorKit
using TensorKitSectors, TensorKit
using Test, TestExtras
using Random
using LinearAlgebra: LinearAlgebra

const MTK = MultiTensorKit
const TK = TensorKit

@isdefined(TestSetup) || include("setup.jl")
using .TestSetup

I = A4Object
Istr = TensorKitSectors.type_repr(I)
r = size(I)

println("----------------------")
println("|    Sector tests    |")
println("----------------------")

@testset "$Istr Basic type properties" verbose = true begin
    @test eval(Meta.parse(sprint(show, I))) == I
    @test eval(Meta.parse(TensorKitSectors.type_repr(I))) == I
end

@testset "$Istr: Value iterator" begin
    @test eltype(values(I)) == I
    @test_throws ArgumentError unit(I)
    sprev = I(1, 1, 1) # first in SectorValues
    for (i, s) in enumerate(values(I))
        @test !isless(s, sprev) # confirm compatibility with sort order
        @test s == @constinferred (values(I)[i])
        @test findindex(values(I), s) == i
        sprev = s
        i >= 10 && break
    end
    @test I(1, 1, 1) == first(values(I))
    @test (@constinferred findindex(values(I), I(1, 1, 1))) == 1
    for s in collect(values(I))
        @test (@constinferred values(I)[findindex(values(I), s)]) == s
    end
end

@testset "$Istr ($i, $j) basic properties" for i in 1:r, j in 1:r
    Cii_obs = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
    Cij_obs = I.(i, j, MTK._get_dual_cache(I)[2][i, j])
    Cji_obs = I.(j, i, MTK._get_dual_cache(I)[2][j, i])
    Cjj_obs = I.(j, j, MTK._get_dual_cache(I)[2][j, j])
    c, m, mop, d = rand(Cii_obs), rand(Cij_obs), rand(Cji_obs), rand(Cjj_obs)

    if i == j
        @testset "Basic fusion properties" begin
            s = rand(Cii_obs, 3)
            @test eval(Meta.parse(sprint(show, s[1]))) == s[1]
            @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
            @test isunit(@constinferred(unit(s[1])))
            u = I.(i, i, MTK._get_dual_cache(I)[1][i])
            @test u == @constinferred(leftunit(u)) == @constinferred(rightunit(u)) ==
                @constinferred(unit(u))
            @test isunit(@constinferred(unit(s[1])))
            @constinferred dual(s[1])
            @test dual(s[1]) == I.(i, i, MTK._get_dual_cache(I)[2][i, i][s[1].label])
            @constinferred dim(s[1])
            @constinferred frobeniusschur(s[1])
            @constinferred Bsymbol(s...)
            @constinferred Fsymbol(s..., s...)
        end
    else
        @testset "Basic module properties" begin
            @test eval(Meta.parse(sprint(show, m))) == m
            @test @constinferred(hash(m)) == hash(deepcopy(m))

            @test isunit(m) == false
            @test isunit(mop) == false
            @test (isunit(@constinferred(leftunit(m))) && isunit(@constinferred(rightunit(m))))
            @test unit(c) == leftunit(m) == rightunit(mop)
            @test unit(d) == rightunit(m) == leftunit(mop)
            @test_throws DomainError unit(m)
            @test_throws DomainError unit(mop)

            @constinferred dual(m)
            @test dual(m) == I.(j, i, MTK._get_dual_cache(I)[2][i, j][m.label])
            @test dual(dual(m)) == m

            @constinferred dim(m)
            @constinferred frobeniusschur(m)
            @constinferred Bsymbol(m, mop, c)
            @constinferred Fsymbol(mop, m, mop, mop, d, c)
        end

        @testset "$Istr Fusion rules" begin
            argerr = ArgumentError("invalid fusion channel")
            # forbidden fusions
            for obs in [(c, d), (d, c), (m, m), (mop, mop), (d, m), (m, c), (mop, d), (c, mop)]
                @test_throws AssertionError("a.j == b.i") isempty(⊗(obs...))
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

for i in 1:r, j in 1:r
    @testset "Unitarity of $Istr F-move ($i, $j)" begin
        if i == j
            @testset "Unitarity of fusion F-move ($i, $j)" begin
                fusion_objects = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
                @test unitarity_test(fusion_objects, fusion_objects, fusion_objects)
            end
        end

        i != j || continue # do this part only when off-diagonal
        mod_objects = I.(i, j, MTK._get_dual_cache(I)[2][i, j])
        left_fusion_objects = I.(i, i, MTK._get_dual_cache(I)[2][i, i])
        right_fusion_objects = I.(j, j, MTK._get_dual_cache(I)[2][j, j])

        # C x C x M -> M or D x D x Mop -> Mop
        @testset "Unitarity of left module F-move ($i, $j)" begin
            @test unitarity_test(left_fusion_objects, left_fusion_objects, mod_objects)
        end

        # M x D x D -> M or Mop x C x C -> Mop
        @testset "Unitarity of right module F-move ($i, $j)" begin
            @test unitarity_test(mod_objects, right_fusion_objects, right_fusion_objects)
        end

        # C x M x D -> M or D x Mop x C -> Mop
        @testset "Unitarity of bimodule F-move ($i, $j)" begin
            @test unitarity_test(left_fusion_objects, mod_objects, right_fusion_objects)
        end

        @testset "Unitarity of mixed module F-move ($i, $j) and opposite ($j, $i)" begin
            modop_objects = I.(j, i, MTK._get_dual_cache(I)[2][j, i])

            # C x M x Mop -> C or D x Mop x M -> D
            @test unitarity_test(left_fusion_objects, mod_objects, modop_objects)
            # M x Mop x C -> C or Mop x M x D -> D
            @test unitarity_test(mod_objects, modop_objects, left_fusion_objects)
            # Mop x C x M -> D or M x D x Mop -> C
            @test unitarity_test(modop_objects, left_fusion_objects, mod_objects)

            # M x Mop x M -> M or Mop x M x Mop -> Mop
            @test unitarity_test(mod_objects, modop_objects, mod_objects)
        end
    end
end

@testset "Triangle equation" begin
    objects = collect(values(I))
    for a in objects, b in objects
        a.j == b.i || continue # skip if not compatible
        @test triangle_equation(a, b; atol=1e-12, rtol=1e-12)
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
                    @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
                end
            end
        end
    end
end

### start of TensorKit tests ###

println("---------------------------------")
println("|    Multifusion space tests    |")
println("---------------------------------")

@timedtestset "Multifusion spaces " verbose = true begin
    @timedtestset "GradedSpace: $(TK.type_repr(Vect[I]))" begin
        gen = (values(I)[k] => (k + 1) for k in 1:length(values(I)))

        V = GradedSpace(gen)
        @test eval(Meta.parse(type_repr(typeof(V)))) == typeof(V)
        @test eval_show(V) == V
        @test eval_show(V') == V'
        @test V' == GradedSpace(gen; dual = true)
        @test V == @constinferred GradedSpace(gen...)
        @test V' == @constinferred GradedSpace(gen...; dual = true)
        @test V == @constinferred GradedSpace(tuple(gen...))
        @test V' == @constinferred GradedSpace(tuple(gen...); dual = true)
        @test V == @constinferred GradedSpace(Dict(gen))
        @test V' == @constinferred GradedSpace(Dict(gen); dual = true)
        @test V == @inferred Vect[I](gen)
        @test V' == @constinferred Vect[I](gen; dual = true)
        @test V == @constinferred Vect[I](gen...)
        @test V' == @constinferred Vect[I](gen...; dual = true)
        @test V == @constinferred Vect[I](Dict(gen))
        @test V' == @constinferred Vect[I](Dict(gen); dual = true)
        @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
        @test V == GradedSpace(reverse(collect(gen))...)
        @test eval_show(V) == V
        @test eval_show(typeof(V)) == typeof(V)

        @test dim(@constinferred(zerospace(V))) == 0

        W = @constinferred GradedSpace(unit => 1 for unit in allunits(I))
        dict = Dict(unit => 1 for unit in allunits(I))
        @test W == GradedSpace(dict)
        @test W == GradedSpace(push!(dict, randsector(I) => 0))
        @test @constinferred(zerospace(V)) == GradedSpace(unit => 0 for unit in allunits(I))
        randunit = rand(collect(allunits(I)))
        @test_throws ArgumentError("Sector $(randunit) appears multiple times") GradedSpace(randunit => 1, randunit => 3)

        @test isunitspace(W)
        @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
        @test_throws ArgumentError leftunitspace(V)
        @test_throws ArgumentError rightunitspace(V)
        @test eval_show(W) == W

        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test isa(InnerProductStyle(V), HasInnerProduct)
        @test isa(InnerProductStyle(V), EuclideanInnerProduct)
        @test isa(V, GradedSpace)
        @test isa(V, GradedSpace{I})
        @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V)) != V
        @test @constinferred(field(V)) == ℂ
        @test @constinferred(sectortype(V)) == I
        slist = @constinferred sectors(V)
        @test @constinferred(hassector(V, first(slist)))
        @test @constinferred(dim(V)) == sum(dim(s) * dim(V, s) for s in slist)
        @test @constinferred(reduceddim(V)) == sum(dim(V, s) for s in slist)
        @constinferred dim(V, first(slist))

        @test @constinferred(⊕(V, zerospace(V))) == V
        @test @constinferred(⊕(V, V)) == Vect[I](c => 2dim(V, c) for c in sectors(V))
        @test @constinferred(⊕(V, V, V, V)) == Vect[I](c => 4dim(V, c) for c in sectors(V))
        @test @constinferred(⊕(V, unitspace(V))) == Vect[I](c => isunit(c) + dim(V, c) for c in sectors(V))
        @test @constinferred(fuse(V, unitspace(V))) == V

        @testset "$Istr ($i, $j) spaces" for i in 1:r, j in 1:r #TODO: look at these tests better
            # space with a single sector
            Wleft = @constinferred Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i))
            Wright = @constinferred Vect[I]((j, j, label) => 1 for label in 1:MTK._numlabels(I, j, j))
            WM = @constinferred Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j))
            WMop = @constinferred Vect[I]((j, i, label) => 1 for label in 1:MTK._numlabels(I, j, i))

            @test leftunitspace(Wleft) == rightunitspace(Wleft)
            @test leftunitspace(Wright) == rightunitspace(Wright)
            @test @constinferred(leftunitspace(⊕(Wleft, WM))) == leftunitspace(Wleft)
            @test @constinferred(leftunitspace(⊕(Wright, WMop))) == leftunitspace(Wright)
            @test @constinferred(rightunitspace(⊕(Wright, WM))) == rightunitspace(Wright)
            @test @constinferred(rightunitspace(⊕(Wleft, WMop))) == rightunitspace(Wleft)

            if i != j # some tests specialised for modules

                # sensible direct sums and fuses
                ul, ur = unit(I(i, i, 1)), unit(I(j, j, 1))
                @test @constinferred(⊕(Wleft, WM)) ==
                    Vect[I](c => 1 for c in sectors(V) if leftunit(c) == ul == rightunit(c) || (c.i == i && c.j == j))
                @test @constinferred(⊕(Wright, WMop)) ==
                    Vect[I](c => 1 for c in sectors(V) if leftunit(c) == ur == rightunit(c) || (c.i == j && c.j == i))
                @test @constinferred(⊕(Wright, WM)) ==
                    Vect[I](c => 1 for c in sectors(V) if rightunit(c) == ur == leftunit(c) || (c.i == i && c.j == j))
                @test @constinferred(⊕(Wleft, WMop)) ==
                    Vect[I](c => 1 for c in sectors(V) if rightunit(c) == ul == leftunit(c) || (c.i == j && c.j == i))
                # round needed below because of numerical F-symbols not being integer when they should be
                # although this test might be stupid, because I'm assuming integer qdims bc everything's a group or irrep on the diagonal
                @test @constinferred(fuse(Wleft, WM)) == Vect[I](c => round(Int, dim(Wleft)) for c in sectors(WM)) # this might be wrong
                @test @constinferred(fuse(Wright, WMop)) == Vect[I](c => round(Int, dim(Wright)) for c in sectors(WMop)) # same

                # less sensible fuse
                @test @constinferred(fuse(Wleft, WMop)) == fuse(Wright, WM) ==
                    Vect[I](c => 0 for c in sectors(V))

                for W in [WM, WMop, Wright]
                    @test infimum(Wleft, W) == Vect[I](c => 0 for c in sectors(V))
                end
            else
                @test @constinferred(⊕(Wleft, Wright)) ==
                    Vect[I](c => 2 for c in sectors(V) if c.i == c.j == i)
                @test @constinferred(fuse(Wleft, WMop)) == fuse(Wright, WM)
            end

            for W in [Wleft, Wright]
                @test @constinferred(⊕(W, rightunitspace(W))) ==
                    Vect[I](c => isunit(c) + dim(W, c) for c in sectors(W))
                @test @constinferred(fuse(W, rightunitspace(W))) == W
            end
        end

        d = Dict{I,Int}()
        for a in sectors(V), b in sectors(V)
            a.j == b.i || continue # skip if not compatible
            for c in a ⊗ b
                d[c] = get(d, c, 0) + dim(V, a) * dim(V, b) * Nsymbol(a, b, c)
            end
        end
        @test @constinferred(fuse(V, V)) == GradedSpace(d)
        @test @constinferred(flip(V)) ==
              Vect[I](conj(c) => dim(V, c) for c in sectors(V))'
        @test flip(V) ≅ V
        @test flip(V) ≾ V
        @test flip(V) ≿ V
        @test @constinferred(⊕(V, V)) == @constinferred supremum(V, ⊕(V, V))
        @test V == @constinferred infimum(V, ⊕(V, V))
        @test V ≺ ⊕(V, V)
        @test !(V ≻ ⊕(V, V))

        randlen = rand(1:length(values(I)))
        s = rand(collect(values(I))[randlen:end]) # such that dim(V, s) > randlen
        @test infimum(V, GradedSpace(s => randlen)) == GradedSpace(s => randlen)
        @test_throws SpaceMismatch (⊕(V, V'))
    end

    @timedtestset "HomSpace with $(TK.type_repr(Vect[I])) involving ($i, $j)" for i in 1:r, j in 1:r
        V1, V2, V3, V4, V5 = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
        Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)),
        Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)), # same as V1
        Vect[I]((i, j, 1) => 3),
        Vect[I]((j, j, label) => 1 for label in 1:MTK._numlabels(I, j, j)))
        W = HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)

        @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
        @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
        @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
        @test eval(Meta.parse(sprint(show, W))) == W
        @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
        @test spacetype(W) == typeof(V1)
        @test sectortype(W) == sectortype(V1)
        @test W[1] == V1
        @test W[2] == V2
        @test W[3] == V3'
        @test W[4] == V4'
        @test W[5] == V5'

        @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
        @test W == deepcopy(W)
        @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
        @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
        @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TK.compose(W, W')

        @test (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ rightunitspace(V5)) ==
            @constinferred(insertleftunit(W)) ==
            @constinferred(insertrightunit(W))
        @test @constinferred(removeunit(insertleftunit(W), $(numind(W) + 1))) == W
        @test_throws BoundsError insertrightunit(W, 6)
        @test_throws BoundsError insertleftunit(W, 0)

        @test (V1 ⊗ V2 ⊗ rightunitspace(V2) ← V3 ⊗ V4 ⊗ V5) ==
                @constinferred(insertrightunit(W, 2))
        @test (V1 ⊗ V2 ← leftunitspace(V3) ⊗ V3 ⊗ V4 ⊗ V5) ==
                @constinferred(insertleftunit(W, 3))
        @test @constinferred(removeunit(insertleftunit(W, 3), 3)) == W
        @test_throws ArgumentError @constinferred(insertrightunit(one(V1) ← V1, 0)) # should I specify it's the other error?
        @test_throws ArgumentError insertleftunit(one(V1) ← V1, 0)
    end
end

println("---------------------------------------")
println("|    Multifusion fusion tree tests    |")
println("---------------------------------------")

@timedtestset "Fusion trees for $(TK.type_repr(I)) involving ($i, $j)" verbose = true for i in 1:r, j in 1:r
    N = 6
    out = random_fusion(I, i, j, Val(N))
    isdual = ntuple(n -> rand(Bool), N)
    in = rand(collect(⊗(out...))) # will be in 𝒞ⱼⱼ with this choice of out

    numtrees = length(fusiontrees(out, in, isdual)) # will be 1 for i != j
    @test numtrees == count(n -> true, fusiontrees(out, in, isdual))

    it = @constinferred fusiontrees(out, in, isdual)
    @constinferred Nothing iterate(it)
    f, s = iterate(it)
    @constinferred Nothing iterate(it, s)
    @test f == @constinferred first(it)
    @testset "Fusion tree $Istr: printing" begin
        @test eval(Meta.parse(sprint(show, f))) == f
    end

    C0, D0 = unit(I(i, i, 1)), unit(I(j, j, 1))
    @testset "Fusion tree $Istr: constructor properties" for u in (C0, D0)
        @constinferred FusionTree((), u, (), (), ())
        @constinferred FusionTree((u,), u, (false,), (), ())
        @constinferred FusionTree((u, u), u, (false, false), (), (1,))
        @constinferred FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @constinferred FusionTree(
            (u, u, u, u), u, (false, false, false, false), (u, u), (1, 1, 1)
        )
        @test_throws MethodError FusionTree((u, u, u), u, (false, false), (u,), (1, 1))
        @test_throws MethodError FusionTree(
            (u, u, u), u, (false, false, false), (u, u), (1, 1)
        )
        @test_throws MethodError FusionTree(
            (u, u, u), u, (false, false, false), (u,), (1, 1, 1)
        )
        @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (), (1,))

        f = FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @test sectortype(f) == I
        @test length(f) == 3
        @test FusionStyle(f) == FusionStyle(I)
        @test BraidingStyle(f) == BraidingStyle(I)

        if FusionStyle(I) isa UniqueFusion
            @constinferred FusionTree((), u, ())
            @constinferred FusionTree((u,), u, (false,))
            @constinferred FusionTree((u, u), u, (false, false))
            @constinferred FusionTree((u, u, u), u)
            if UnitStyle(I) isa SimpleUnit
                @constinferred FusionTree((u, u, u, u))
            else
                @test_throws ArgumentError FusionTree((u, u, u, u))
            end
            @test_throws MethodError FusionTree((u, u), u, (false, false, false))
        else
            @test_throws ArgumentError FusionTree((), u, ())
            @test_throws ArgumentError FusionTree((u,), u, (false,))
            @test_throws ArgumentError FusionTree((u, u), u, (false, false))
            @test_throws ArgumentError FusionTree((u, u, u), u)
            if I <: ProductSector && UnitStyle(I) isa GenericUnit
                @test_throws DomainError FusionTree((u, u, u, u))
            else
                @test_throws ArgumentError FusionTree((u, u, u, u))
            end
        end
    end

    @testset "Fusion tree $Istr: insertat" begin
        N = 4
        out2 = random_fusion(I, i, j, Val(N))
        in2 = rand(collect(⊗(out2...)))
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for k in 1:N
            out1 = random_fusion(I, i, j, Val(N)) # guaranteed good fusion
            out1 = Base.setindex(out1, in2, i) # can lead to poor fusion
            while isempty(⊗(out1...)) # TODO: better way to do this?
                out1 = random_fusion(I, i, j, Val(N))
                out1 = Base.setindex(out1, in2, i)
            end
            in1 = rand(collect(⊗(out1...)))
            isdual1 = ntuple(n -> rand(Bool), N)
            isdual1 = Base.setindex(isdual1, false, k)
            f1 = rand(collect(fusiontrees(out1, in1, isdual1)))

            trees = @constinferred TK.insertat(f1, k, f2)
            @test norm(values(trees)) ≈ 1

            f1a, f1b = @constinferred TK.split(f1, $k)
            @test length(TK.insertat(f1b, 1, f1a)) == 1
            @test first(TK.insertat(f1b, 1, f1a)) == (f1 => 1)

            # no braid tests for non-hardcoded example
        end
    end
    # no planar trace tests
    
    @testset "Fusion tree $Istr: elementary artin braid" begin
        N = length(out)
        isdual = ntuple(n -> rand(Bool), N)
        # no general artin braid test

        # not sure how useful this test is, it does the trivial braiding (choice of out)
        f = rand(collect(it)) # in this case the 1 tree
        d1 = TK.artin_braid(f, 2) # takes unit C0 with current out
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3; inv=true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 2; inv=true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        for (f1, coeff1) in d1
            if f1 == f
                @test coeff1 ≈ 1
            else
                @test isapprox(coeff1, 0; atol=1.0e-12, rtol=1.0e-12)
            end
        end
    end

    # no braiding and permuting test
    @testset "Fusion tree $Istr: merging" begin
        N = 3
        out1 = random_fusion(I, i, j, N)
        out2 = random_fusion(I, i, j, N)
        in1 = rand(collect(⊗(out1...)))
        in2 = rand(collect(⊗(out2...)))
        tp = ⊗(in1, in2) # messy solution but it works
        while isempty(tp)
            out1 = random_fusion(I, i, j, Val(N))
            out2 = random_fusion(I, i, j, Val(N))
            in1 = rand(collect(⊗(out1...)))
            in2 = rand(collect(⊗(out2...)))
            tp = ⊗(in1, in2)
        end

        f1 = rand(collect(fusiontrees(out1, in1)))
        f2 = rand(collect(fusiontrees(out2, in2)))


        @test dim(in1) * dim(in2) ≈ sum(abs2(coeff) * dim(c) for c in in1 ⊗ in2
                                        for μ in 1:Nsymbol(in1, in2, c)
                                        for (f, coeff) in TK.merge(f1, f2, c, μ))
        # no merge and braid interplay tests
    end

    # double fusion tree tests
    N = 4
    out = random_fusion(I, i, j, Val(N))
    out2 = random_fusion(I, i, j, Val(N))
    tp = ⊗(out...)
    tp2 = ⊗(out2...)
    while isempty(intersect(tp, tp2)) # guarantee fusion to same coloring
        out2 = random_fusion(I, i, j, Val(N))
        tp2 = ⊗(out2...)
    end
    @test_throws ArgumentError fusiontrees((out..., map(dual, out)...))
    incoming = rand(collect(intersect(tp, tp2)))
    f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
    f2 = rand(collect(fusiontrees(out2, incoming, ntuple(n -> rand(Bool), N)))) # no permuting

    @testset "Double fusion tree $Istr: repartitioning" begin
        for n in 0:(2 * N)
            d = @constinferred TK.repartition(f1, f2, $n)
            @test dim(incoming) ≈
                  sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
            d2 = Dict{typeof((f1, f2)),valtype(d)}()
            for ((f1′, f2′), coeff) in d
                for ((f1′′, f2′′), coeff2) in TK.repartition(f1′, f2′, N)
                    d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                end
            end
            for ((f1′, f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test isapprox(coeff2, 0; atol=1.0e-12, rtol=1.0e-12)
                end
            end
        end
    end

    # no double fusion tree permutation tests

    # very slow for (1, 6), (3, 4), (3, 5), (3, 6), (5, 6), (6, 1), (6, 5), (7, 1), (7, 4), (7, 6)
    @testset "Double fusion tree $Istr: transposition" begin
        for n in 0:(2N)
            i0 = rand(1:(2N))
            p = mod1.(i0 .+ (1:(2N)), 2N)
            ip = mod1.(-i0 .+ (1:(2N)), 2N)
            p′ = tuple(getindex.(Ref(vcat(1:N, (2N):-1:(N + 1))), p)...)
            p1, p2 = p′[1:n], p′[(2N):-1:(n + 1)]
            ip′ = tuple(getindex.(Ref(vcat(1:n, (2N):-1:(n + 1))), ip)...)
            ip1, ip2 = ip′[1:N], ip′[(2N):-1:(N + 1)]

            d = @constinferred transpose(f1, f2, p1, p2)
            @test dim(incoming) ≈
                  sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
            d2 = Dict{typeof((f1, f2)),valtype(d)}()
            for ((f1′, f2′), coeff) in d
                d′ = transpose(f1′, f2′, ip1, ip2)
                for ((f1′′, f2′′), coeff2) in d′
                    d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                end
            end
            for ((f1′, f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test abs(coeff2) < 1.0e-12
                end
            end
        end
    end
    
    @testset "Double fusion tree $Istr: planar trace" begin
        d1 = transpose(f1, f1, (N + 1, 1:N..., ((2N):-1:(N + 3))...), (N + 2,))
        f1front, = TK.split(f1, N - 1)
        T = sectorscalartype(I)
        d2 = Dict{typeof((f1front, f1front)),T}()
        for ((f1′, f2′), coeff′) in d1
            for ((f1′′, f2′′), coeff′′) in
                TK.planar_trace(f1′, f2′, (2:N...,), (1, ((2N):-1:(N + 3))...), (N + 1,),
                                (N + 2,))
                coeff = coeff′ * coeff′′
                d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff
            end
        end
        for ((f1_, f2_), coeff) in d2
            if (f1_, f2_) == (f1front, f1front)
                @test coeff ≈ dim(f1.coupled) / dim(f1front.coupled)
            else
                @test abs(coeff) < 1.0e-12
            end
        end
    end
end

println("-------------------------------------------")
println("|    Multifusion diagonal tensor tests    |")
println("-------------------------------------------")

V = Vect[I](values(I)[k] => 1 for k in 1:length(values(I)))

@timedtestset "DiagonalTensor" begin
    @timedtestset "Basic properties and algebra" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            # constructors
            t = @constinferred DiagonalTensorMap{T}(undef, V)
            t = @constinferred DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, space(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2 ← V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, domain(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2)
            # properties
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test scalartype(t) == T
            @test codomain(t) == ProductSpace(V)
            @test domain(t) == ProductSpace(V)
            @test space(t) == (V ← V)
            @test space(t') == (V ← V)
            @test dim(t) == dim(space(t))
            # blocks
            bs = @constinferred blocks(t)
            (c, b1), state = @constinferred Nothing iterate(bs)
            @test c == first(blocksectors(V ← V))
            next = @constinferred Nothing iterate(bs, state)
            b2 = @constinferred block(t, first(blocksectors(t)))
            @test b1 == b2
            @test eltype(bs) === Pair{typeof(c), typeof(b1)}
            @test typeof(b1) === TensorKit.blocktype(t)
            # basic linear algebra
            @test isa(@constinferred(norm(t)), real(T))
            @test norm(t)^2 ≈ dot(t, t)
            α = rand(T)
            @test norm(α * t) ≈ abs(α) * norm(t)
            @test norm(t + t, 2) ≈ 2 * norm(t, 2)
            @test norm(t + t, 1) ≈ 2 * norm(t, 1)
            @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
            p = 3 * rand(Float64)
            @test norm(t + t, p) ≈ 2 * norm(t, p)
            @test norm(t) ≈ norm(t')

            @test t == @constinferred(TensorMap(t))
            @test norm(t + TensorMap(t)) ≈ 2 * norm(t)

            @test norm(zerovector!(t)) == 0
            @test norm(one!(t)) ≈ sqrt(dim(V))
            @test one!(t) == id(V)
            if T != BigFloat # seems broken for now
                @test norm(one!(t) - id(V)) == 0
            end

            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t3 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            α = rand(T)
            β = rand(T)
            @test @constinferred(dot(t1, t2)) ≈ conj(dot(t2, t1))
            @test dot(t2, t1) ≈ conj(dot(t2', t1'))
            @test dot(t3, α * t1 + β * t2) ≈ α * dot(t3, t1) + β * dot(t3, t2)
        end
    end

    @timedtestset "Basic linear algebra: test via conversion" begin
        for T in (Float32, ComplexF64)
            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @test norm(t1, 2) ≈ norm(convert(TensorMap, t1), 2)
            @test dot(t2, t1) ≈ dot(convert(TensorMap, t2), convert(TensorMap, t1))
            α = rand(T)
            @test convert(TensorMap, α * t1) ≈ α * convert(TensorMap, t1)
            @test convert(TensorMap, t1') ≈ convert(TensorMap, t1)'
            @test convert(TensorMap, t1 + t2) ≈ convert(TensorMap, t1) + convert(TensorMap, t2)
        end
    end
    @timedtestset "Real and imaginary parts" begin
        for T in (Float64, ComplexF64, ComplexF32)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)

            tr = @constinferred real(t)
            @test scalartype(tr) <: Real
            @test real(convert(TensorMap, t)) == convert(TensorMap, tr)

            ti = @constinferred imag(t)
            @test scalartype(ti) <: Real
            @test imag(convert(TensorMap, t)) == convert(TensorMap, ti)

            tc = @inferred complex(t)
            @test scalartype(tc) <: Complex
            @test complex(convert(TensorMap, t)) == convert(TensorMap, tc)

            tc2 = @inferred complex(tr, ti)
            @test tc2 ≈ tc
        end
    end
    @timedtestset "Tensor conversion" begin
        t = @constinferred DiagonalTensorMap(undef, V)
        rand!(t.data)
        # element type conversion
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        # to and from generic TensorMap
        td = DiagonalTensorMap(TensorMap(t))
        @test t == td
        @test typeof(td) == typeof(t)
    end
    @timedtestset "Trace, Multiplication and inverse" begin
        t1 = DiagonalTensorMap(rand(Float64, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        @test tr(TensorMap(t1)) == @constinferred tr(t1)
        @test tr(TensorMap(t2)) == @constinferred tr(t2)
        @test TensorMap(@constinferred t1 * t2) ≈ TensorMap(t1) * TensorMap(t2)
        @test TensorMap(@constinferred t1 \ t2) ≈ TensorMap(t1) \ TensorMap(t2)
        @test TensorMap(@constinferred t1 / t2) ≈ TensorMap(t1) / TensorMap(t2)
        @test TensorMap(@constinferred inv(t1)) ≈ inv(TensorMap(t1))
        @test TensorMap(@constinferred pinv(t1)) ≈ pinv(TensorMap(t1))
        @test all(
            Base.Fix2(isa, DiagonalTensorMap), (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1))
        )
        # no V * V' * V ← V or V^2 ← V tests due to Nsymbol erroring where fusion is forbidden
    end
    @timedtestset "Tensor contraction " for i in 1:r
        W = Vect[I]((i, i, label) => 2 for label in 1:MTK._numlabels(I, i, i))

        d = DiagonalTensorMap(rand(ComplexF64, reduceddim(W)), W)
        t = TensorMap(d)
        A = randn(ComplexF64, W ⊗ W' ⊗ W, W)
        B = randn(ComplexF64, W ⊗ W' ⊗ W, W ⊗ W') # empty for modules so untested

        @planar E1[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * d[1; -4]
        @planar E2[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * t[1; -4]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * d'[-5; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * t'[-5; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * d[-1; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * t[-1; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * d[1; -2]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * t[1; -2]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * d'[-3; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * t'[-3; 1]
        @test E1 ≈ E2
    end
    @timedtestset "Tensor functions" begin
        for T in (Float64, ComplexF64)
            d = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            # rand is important for positive numbers in the real case, for log and sqrt
            t = TensorMap(d)
            @test @constinferred exp(d) ≈ exp(t)
            @test @constinferred log(d) ≈ log(t)
            @test @constinferred sqrt(d) ≈ sqrt(t)
            @test @constinferred sin(d) ≈ sin(t)
            @test @constinferred cos(d) ≈ cos(t)
            @test @constinferred tan(d) ≈ tan(t)
            @test @constinferred cot(d) ≈ cot(t)
            @test @constinferred sinh(d) ≈ sinh(t)
            @test @constinferred cosh(d) ≈ cosh(t)
            @test @constinferred tanh(d) ≈ tanh(t)
            @test @constinferred coth(d) ≈ coth(t)
            @test @constinferred asin(d) ≈ asin(t)
            @test @constinferred acos(d) ≈ acos(t)
            @test @constinferred atan(d) ≈ atan(t)
            @test @constinferred acot(d) ≈ acot(t)
            @test @constinferred asinh(d) ≈ asinh(t)
            @test @constinferred acosh(one(d) + d) ≈ acosh(one(t) + t)
            @test @constinferred atanh(d) ≈ atanh(t)
            @test @constinferred acoth(one(t) + d) ≈ acoth(one(d) + t)
        end
    end
end

# no conversion tests because no fusion tensor
# no permute tests: NoBraiding()

println("---------------------------------------")
println("Tensors with symmetry: $Istr")
println("---------------------------------------")

@timedtestset "Tensors with symmetry involving $Istr ($i, $j)" verbose = true for i in 1:r, j in 1:r
    isdiag = i == j

    VC = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
                Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),   # avoids OOMs?
                Vect[I](unit(I(i, i, 1)) => 2, rand_object(I, i, i) => 1),
                Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
                Vect[I](unit(I(i, i, 1)) => 2, rand_object(I, i, i) => 3)
        )

    VM = Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)) # all module objects 

    VM1 = (Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1), # written so V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 works
            Vect[I](rand_object(I, i, j) => 2), # generally less blocksectors
            Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            VM, # important that V4 is module-graded
            Vect[I](unit(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    VM2 = (Vect[I](rand_object(I, i, j) => 2), # second set where module is V1 here
            Vect[I](unit(I(j, j, 1)) => 1, rand_object(I, j, j) => 1),
            Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            VM,
            Vect[I](unit(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    Vcol = isdiag ? (VC,) : (VM1, VM2)  # avoid duplicate runs

    for V in Vcol # TODO: add enumerate to keep track of potential erroring space
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = isdiag ? V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5 : V3 ⊗ V4 ⊗ V5 # fusion matters
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
                t = @constinferred zeros(T, W) # empty for i != j b/c blocks are module-graded
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), length(W), 0, Vector{T}}
                # blocks
                bs = @constinferred blocks(t)
                if !isempty(bs)
                    (c, b1), state = @constinferred Nothing iterate(bs) # errors if fusion gives empty data
                    # @test c == first(blocksectors(W)) # unit doesn't have label 1
                    next = @constinferred Nothing iterate(bs, state)
                    b2 = @constinferred block(t, first(blocksectors(t)))
                    @test b1 == b2
                    @test eltype(bs) === Pair{typeof(c),typeof(b1)}
                    @test typeof(b1) === TK.blocktype(t)
                    @test typeof(c) === sectortype(t)
                end
            end
        end

        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 # rewritten to be compatible with module fusion
            for T in (Int, Float32, ComplexF64)
                t = @constinferred rand(T, W)
                d = convert(Dict, t)
                @test t == convert(TensorMap, d)
            end
        end

        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W) # fusion matters here
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                # blocks for adjoint
                bs = @constinferred blocks(t')
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W'))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t', first(blocksectors(t')))
                @test b1 == b2
                @test eltype(bs) === Pair{typeof(c), typeof(b1)}
                @test typeof(b1) === TensorKit.blocktype(t')
                @test typeof(c) === sectortype(t)
                # linear algebra
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                p = 3 * rand(Float64)
                @test norm(t + t, p) ≈ 2 * norm(t, p)
                @test norm(t) ≈ norm(t')

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                if !isempty(blocksectors(V2 ⊗ V1))
                    i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1)) # can't reverse fusion here when modules are involved
                    i2 = @constinferred(isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2))
                    @test i1 * i2 == @constinferred(id(T, V1 ⊗ V2))
                    @test i2 * i1 == @constinferred(id(Vector{T}, V2 ⊗ V1))
                end

                w = @constinferred isometry(T, V1 ⊗ (rightunitspace(V1) ⊕ rightunitspace(V1)), V1)
                @test dim(w) == 2 * dim(V1 ← V1)
                @test w' * w == id(Vector{T}, V1)
                @test w * w' == (w * w')^2
            end
        end

        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W) # fusion matters here
                t2 = @constinferred insertleftunit(t)
                @test t2 == @constinferred insertrightunit(t)
                @test space(t2) == insertleftunit(space(t))
                @test @constinferred(removeunit(t2, $(numind(t2)))) == t
                t3 = @constinferred insertleftunit(t; copy = true)
                @test t3 == @constinferred insertrightunit(t; copy = true)
                @test @constinferred(removeunit(t3, $(numind(t3)))) == t

                @test numind(t2) == numind(t) + 1
                @test scalartype(t2) === T
                @test t.data === t2.data

                @test t.data !== t3.data
                for (c, b) in blocks(t)
                    @test b == block(t3, c)
                end

                t4 = @constinferred insertrightunit(t, 3; dual = true)
                @test numin(t4) == numin(t) + 1 && numout(t4) == numout(t)
                for (c, b) in blocks(t)
                    @test b == block(t4, c)
                end
                @test @constinferred(removeunit(t4, 4)) == t

                t5 = @constinferred insertleftunit(t, 4; dual = true)
                @test numin(t5) == numin(t) + 1 && numout(t5) == numout(t)
                for (c, b) in blocks(t)
                    @test b == block(t5, c)
                end
                @test @constinferred(removeunit(t5, 4)) == t
            end
        end

        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred randn(W ← W) # fusion matters here
            @test typeof(convert(TensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end

        @timedtestset "Full trace: test self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2 ← V1 ⊗ V2) # avoid permutes
            ss = @constinferred tr(t)
            @test conj(ss) ≈ tr(t')
            @planar s2 = t[a b; a b]
            @planar t3[a; b] := t[a c; b c]
            @planar s3 = t3[a; a]

            @test ss ≈ s2
            @test ss ≈ s3
        end

        @timedtestset "Partial trace: test self-consistency" begin
            t = rand(ComplexF64, V3 ⊗ V4 ⊗ V5 ← V3 ⊗ V4 ⊗ V5) # compatible with module fusion
            @planar t2[a; b] := t[c a d; c b d]
            @planar t4[a b; c d] := t[e a b; e c d]
            @planar t5[a; b] := t4[a c; b c]
            @test t2 ≈ t5
        end

        @timedtestset "Trace and contraction" begin #TODO: find some version of this that works for off-diagonal case
            t1 = rand(ComplexF64, V3 ⊗ V4 ⊗ V5)
            t2 = rand(ComplexF64, V3 ⊗ V4 ⊗ V5)
            t3 = t1 ⊗ t2'
            # if all(a.i != a.j for a in blocksectors(t3))
            #     replace!(x -> rand(ComplexF64), t3.data) # otherwise full of zeros in off-diagonal case
            # end
            if all(a.i == a.j for a in blocksectors(t3))
                @planar ta[b; a] := conj(t2[x, a, y]) * t1[x, b, y] # works for diagonal case
                @planar tb[a; b] := t3[x a y; x b y]
                @test ta ≈ tb
            end
        end

        @timedtestset "Multiplication of isometries: test properties" begin
            W2 = V4 ⊗ V5
            W1 = W2 ⊗ (rightunitspace(V5) ⊕ rightunitspace(V5))
            for T in (Float64, ComplexF64)
                t1 = randisometry(T, W1, W2)
                t2 = randisometry(T, W2 ← W2)
                @test isisometric(t1)
                @test isunitary(t2)
                P = t1 * t1'
                @test P * P ≈ P
            end
        end

        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2
            W2 = V3 ⊗ V4 ⊗ V5
            for T in (Float64, ComplexF64)
                t1 = rand(T, W1, W1)
                t2 = rand(T, W2 ← W2)
                t = rand(T, W1, W2)
                @test t1 * (t1 \ t) ≈ t
                @test (t / t2) * t2 ≈ t
                @test t1 \ one(t1) ≈ inv(t1)
                @test one(t1) / t1 ≈ pinv(t1)
                @test_throws SpaceMismatch inv(t)
                @test_throws SpaceMismatch t2 \ t
                @test_throws SpaceMismatch t / t1
                tp = pinv(t) * t
                @test tp ≈ tp * tp
            end
        end

        @timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            t = randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end

        @timedtestset "Sylvester equation" begin
            for T in (Float32, ComplexF64)
                tA = rand(T, V1 ⊗ V2, V1 ⊗ V2) # rewritten for modules
                tB = rand(T, V4 ⊗ V5, V4 ⊗ V5)
                tA = 3 // 2 * leftorth(tA; alg=TK.Polar())[1]
                tB = 1 // 5 * leftorth(tB; alg=TK.Polar())[1]
                tC = rand(T, V1 ⊗ V2, V4 ⊗ V5)
                t = @constinferred sylvester(tA, tB, tC)
                @test codomain(t) == V1 ⊗ V2
                @test domain(t) == V4 ⊗ V5
                @test norm(tA * t + t * tB + tC) <
                    (norm(tA) + norm(tB) + norm(tC)) * eps(real(T))^(2 / 3)
            end
        end

        @timedtestset "Tensor product: test via norm preservation" begin # OOMs over here with full spaces
            for T in (Float32, ComplexF64)
                if !isempty(blocksectors(V2 ⊗ V1))
                    t1 = rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
                    t2 = rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
                else
                    t1 = rand(T, V3 ⊗ V4 ⊗ V5, V1 ⊗ V2)
                    t2 = rand(T, V5' ⊗ V4' ⊗ V3', V2' ⊗ V1')
                end
                t = @constinferred (t1 ⊗ t2)
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end

        # TODO: should this test exist?
        @timedtestset "Tensor product: test via tensor contraction" begin
            # W = V3 ⊗ V4 ⊗ V5 ← V1 ⊗ V2
            W = V4 ← V1 ⊗ V2 # less costly
            for T in (Float32, ComplexF64)
                if !isdiag
                    t1 = rand(T, W)
                    t2 = rand(T, V4' ← V2' ⊗ V1')
                    # t2 = rand(T, V5' ⊗ V4' ⊗ V3', V2' ⊗ V1') # same as previous test
                    # @planar t′[1 2 3 6 7 8; 4 5 9 10] := t1[1 2 3; 4 5] * t2[6 7 8; 9 10]
                    @planar t′[1 4; 2 3 5 6] := t1[1; 2 3] * t2[4; 5 6]
                else
                    t1 = rand(T, V2 ⊗ V3, V1)
                    t2 = rand(T, V2, V1 ⊗ V3)
                    @planar t′[1 2 4; 3 5 6] := t1[1 2; 3] * t2[4; 5 6]
                end
                t = @constinferred (t1 ⊗ t2)
                @test t ≈ t′
            end
        end
    end

    @timedtestset "Tensor absorption" begin
        # absorbing small into large
        if !isempty(blocksectors(V2 ⊗ V3))
            t1 = zeros(V1 ⊕ V1, V2 ⊗ V3)
            t2 = rand(V1, V2 ⊗ V3)
        else
            t1 = zeros(V1 ⊕ V2, V3 ⊗ V4 ⊗ V5)
            t2 = rand(V1, V3 ⊗ V4 ⊗ V5)
        end
        t3 = @constinferred absorb(t1, t2)
        @test norm(t3) ≈ norm(t2)
        @test norm(t1) == 0
        t4 = @constinferred absorb!(t1, t2)
        @test t1 === t4
        @test t3 ≈ t4

        # absorbing large into small
        if !isempty(blocksectors(V2 ⊗ V3))
            t1 = rand(V1 ⊕ V1, V2 ⊗ V3)
            t2 = zeros(V1, V2 ⊗ V3)
        else
            t1 = rand(V1 ⊕ V2, V3 ⊗ V4 ⊗ V5)
            t2 = zeros(V1, V3 ⊗ V4 ⊗ V5)
        end
        t3 = @constinferred absorb(t2, t1)
        @test norm(t3) < norm(t1)
        @test norm(t2) == 0
        t4 = @constinferred absorb!(t2, t1)
        @test t2 === t4
        @test t3 ≈ t4
    end
end

println("---------------------------------------")
println("Factorizations with symmetry: $Istr")
println("---------------------------------------")

@timedtestset "Factorizations with symmetry involving $Istr ($i, $j)" verbose = true for i in 1:r, j in 1:r
    isdiag = i == j

    VC = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
                Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),   # avoids OOMs?
                Vect[I](unit(I(i, i, 1)) => 2, rand_object(I, i, i) => 1),
                Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
                Vect[I](unit(I(i, i, 1)) => 2, rand_object(I, i, i) => 3)
        )

    VM = Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)) # all module objects 

    VM1 = (Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1), # written so V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 works
            Vect[I](rand_object(I, i, j) => 2), # generally less blocksectors
            Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            VM, # important that V4 is module-graded
            Vect[I](unit(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    VM2 = (Vect[I](rand_object(I, i, j) => 2), # second set where module is V1 here
            Vect[I](unit(I(j, j, 1)) => 1, rand_object(I, j, j) => 1),
            Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            VM,
            Vect[I](unit(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    Vs = isdiag ? (VC,) : (VM1, VM2)  # avoid duplicate runs

    # some fail for (2, 2), (3, 3), (6, 6)
    # rightorth RQ(pos) and Polar (fail) for 2nd space
    # leftorth with QL(pos) and Polar for 1st space
    # leftnull QR for 1st space
    # cond and rank leftnull for 1st space

    # factorization tests require equal objects in blocksectors of domain and codomain, so just put them all
    # FIXME: not sure if still needed
    # VC_all = fill(Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)), 5)

    # VM1_all = (Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
    #     VM,
    #     Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
    #     VM,
    #     Vect[I](unit(I(j, j, 1)) => 1, rand_object(I, j, j) => 2)
    # )

    # VM2_all = (VM,
    #     Vect[I](unit(I(j, j, 1)) => 1, rand_object(I, j, j) => 1),
    #     Vect[I](unit(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
    #     VM,
    #     Vect[I](unit(I(j, j, 1)) => 2, rand_object(I, j, j) => 2)
    #     )

    # fact_Vs = (i != j) ? (VM1_all, VM2_all) : (VC_all,)

    @timedtestset "Factorization" for V in Vs
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2
        @assert !isempty(blocksectors(W))
        @assert !isempty(intersect(blocksectors(V4), blocksectors(W)))

        @testset "QR decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                N = @constinferred qr_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes
                t = rand(T, V1 ⊗ V2, zerospace(V1))

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)
                @test dim(R) == dim(t) == 0

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                N = @constinferred qr_null(t)
                @test isunitary(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "LQ decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                Nᴴ = @constinferred lq_null(t)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            for T in eltypes
                # empty tensor
                t = rand(T, zerospace(V1), V1 ⊗ V2)

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)
                @test dim(L) == dim(t) == 0

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                Nᴴ = @constinferred lq_null(t)
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "Polar decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                @assert domain(t) ≾ codomain(t)
                w, p = @constinferred left_polar(t)
                @test w * p ≈ t
                @test isisometric(w)
                @test isposdef(p)

                w, p = @constinferred left_orth(t; alg = :polar)
                @test w * p ≈ t
                @test isisometric(w)
            end

            for T in eltypes,
                    t in (rand(T, W, W), rand(T, W, W)', rand(T, V4, W), rand(T, W, V4)')

                @assert codomain(t) ≾ domain(t)
                p, wᴴ = @constinferred right_polar(t)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
                @test isposdef(p)

                p, wᴴ = @constinferred right_orth(t; alg = :polar)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
            end
        end

        @testset "SVD" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, W, V4), rand(T, V4, W),
                        rand(T, W, V4)', rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                u, s, vᴴ = @constinferred svd_full(t)
                @test u * s * vᴴ ≈ t
                @test isunitary(u)
                @test isunitary(vᴴ)

                u, s, vᴴ = @constinferred svd_compact(t)
                @test u * s * vᴴ ≈ t
                @test isisometric(u)
                @test isposdef(s)
                @test isisometric(vᴴ; side = :right)

                s′ = @constinferred svd_vals(t)
                @test s′ ≈ diagview(s)
                @test s′ isa TensorKit.SectorVector

                v, c = @constinferred left_orth(t; alg = :svd)
                @test v * c ≈ t
                @test isisometric(v)

                c, vᴴ = @constinferred right_orth(t; alg = :svd)
                @test c * vᴴ ≈ t
                @test isisometric(vᴴ; side = :right)

                N = @constinferred left_null(t; alg = :svd)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t; trunc = (; atol = 100 * eps(norm(t))))
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; alg = :svd)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; trunc = (; atol = 100 * eps(norm(t))))
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes, t in (rand(T, W, zerospace(V1)), rand(T, zerospace(V1), W))
                U, S, Vᴴ = @constinferred svd_full(t)
                @test U * S * Vᴴ ≈ t
                @test isunitary(U)
                @test isunitary(Vᴴ)

                U, S, Vᴴ = @constinferred svd_compact(t)
                @test U * S * Vᴴ ≈ t
                @test dim(U) == dim(S) == dim(Vᴴ) == dim(t) == 0
            end
        end

        @testset "truncated SVD" begin
            for T in eltypes,
                    t in (
                        randn(T, W, W), randn(T, W, W)',
                        randn(T, W, V4), randn(T, V4, W),
                        randn(T, W, V4)', randn(T, V4, W)',
                        DiagonalTensorMap(randn(T, reduceddim(V1)), V1),
                    )

                @constinferred normalize!(t)

                U, S, Vᴴ, ϵ = @constinferred svd_trunc(t; trunc = notrunc())
                @test U * S * Vᴴ ≈ t
                @test ϵ ≈ 0
                @test isisometric(U)
                @test isisometric(Vᴴ; side = :right)

                # dimension of S is a float for IsingBimodule
                nvals = round(Int, dim(domain(S)) / 2)
                trunc = truncrank(nvals)
                U1, S1, Vᴴ1, ϵ1 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ1' ≈ U1 * S1
                @test isisometric(U1)
                @test isisometric(Vᴴ1; side = :right)
                @test norm(t - U1 * S1 * Vᴴ1) ≈ ϵ1 atol = eps(real(T))^(4 / 5)
                @test dim(domain(S1)) <= nvals

                λ = minimum(diagview(S1))
                trunc = trunctol(; atol = λ - 10eps(λ))
                U2, S2, Vᴴ2, ϵ2 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ2' ≈ U2 * S2
                @test isisometric(U2)
                @test isisometric(Vᴴ2; side = :right)
                @test norm(t - U2 * S2 * Vᴴ2) ≈ ϵ2 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S1)) >= λ
                @test U2 ≈ U1
                @test S2 ≈ S1
                @test Vᴴ2 ≈ Vᴴ1
                @test ϵ1 ≈ ϵ2

                trunc = truncspace(space(S2, 1))
                U3, S3, Vᴴ3, ϵ3 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ3' ≈ U3 * S3
                @test isisometric(U3)
                @test isisometric(Vᴴ3; side = :right)
                @test norm(t - U3 * S3 * Vᴴ3) ≈ ϵ3 atol = eps(real(T))^(4 / 5)
                @test space(S3, 1) ≾ space(S2, 1)

                for trunc in (truncerror(; atol = ϵ2), truncerror(; rtol = ϵ2 / norm(t)))
                    U4, S4, Vᴴ4, ϵ4 = @constinferred svd_trunc(t; trunc)
                    @test t * Vᴴ4' ≈ U4 * S4
                    @test isisometric(U4)
                    @test isisometric(Vᴴ4; side = :right)
                    @test norm(t - U4 * S4 * Vᴴ4) ≈ ϵ4 atol = eps(real(T))^(4 / 5)
                    @test ϵ4 ≤ ϵ2
                end

                trunc = truncrank(nvals) & trunctol(; atol = λ - 10eps(λ))
                U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ5' ≈ U5 * S5
                @test isisometric(U5)
                @test isisometric(Vᴴ5; side = :right)
                @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S5)) >= λ
                @test dim(domain(S5)) ≤ nvals
            end
        end

        @testset "Eigenvalue decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                d, v = @constinferred eig_full(t)
                @test t * v ≈ v * d

                d′ = @constinferred eig_vals(t)
                @test d′ ≈ diagview(d)
                @test d′ isa TensorKit.SectorVector

                vdv = project_hermitian!(v' * v)
                @test @constinferred isposdef(vdv)
                t isa DiagonalTensorMap || @test !isposdef(t) # unlikely for non-hermitian map

                nvals = round(Int, dim(domain(t)) / 2)
                d, v = @constinferred eig_trunc(t; trunc = truncrank(nvals))
                @test t * v ≈ v * d
                @test dim(domain(d)) ≤ nvals

                t2 = @constinferred project_hermitian(t)
                D, V = eigen(t2)
                @test isisometric(V)
                D̃, Ṽ = @constinferred eigh_full(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(real, diagview(D))
                @test cond(Ṽ) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

                d, v = @constinferred eigh_full(t2)
                @test t2 * v ≈ v * d
                @test isunitary(v)

                d′ = @constinferred eigh_vals(t2)
                @test d′ ≈ diagview(d)
                @test d′ isa TensorKit.SectorVector

                λ = minimum(real, diagview(d))
                @test cond(v) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t) - 0.1 * one(t2))

                d, v = @constinferred eigh_trunc(t2; trunc = truncrank(nvals))
                @test t2 * v ≈ v * d
                @test dim(domain(d)) ≤ nvals
            end
        end

        @testset "Condition number and rank" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, W, V4), rand(T, V4, W),
                        rand(T, W, V4)', rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                d1, d2 = dim(codomain(t)), dim(domain(t))
                r = rank(t)
                @test r == min(d1, d2)
                @test typeof(r) == typeof(d1)
                M = left_null(t)
                @test @constinferred(rank(M)) + r ≈ d1
                Mᴴ = right_null(t)
                @test rank(Mᴴ) + r ≈ d2
            end
            for T in eltypes
                u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                @test @constinferred(cond(u)) ≈ one(real(T))
                @test @constinferred(rank(u)) == dim(V1 ⊗ V2)

                t = rand(T, zerospace(V1), W)
                @test rank(t) == 0
                t2 = rand(T, zerospace(V1) * zerospace(V2), zerospace(V1) * zerospace(V2))
                @test rank(t2) == 0
                @test cond(t2) == 0.0
            end
            for T in eltypes, t in (rand(T, W, W), rand(T, W, W)')
                project_hermitian!(t)
                vals = @constinferred LinearAlgebra.eigvals(t)
                λmax = maximum(s -> maximum(abs, s), values(vals))
                λmin = minimum(s -> minimum(abs, s), values(vals))
                @test cond(t) ≈ λmax / λmin
            end
        end

        @testset "Hermitian projections" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )
                normalize!(t)
                noisefactor = eps(real(T))^(3 / 4)

                th = (t + t') / 2
                ta = (t - t') / 2
                tc = copy(t)

                th′ = @constinferred project_hermitian(t)
                @test ishermitian(th′)
                @test th′ ≈ th
                @test t == tc
                th_approx = th + noisefactor * ta
                @test !ishermitian(th_approx) || (T <: Real && t isa DiagonalTensorMap)
                @test ishermitian(th_approx; atol = 10 * noisefactor)

                ta′ = project_antihermitian(t)
                @test isantihermitian(ta′)
                @test ta′ ≈ ta
                @test t == tc
                ta_approx = ta + noisefactor * th
                @test !isantihermitian(ta_approx)
                @test isantihermitian(ta_approx; atol = 10 * noisefactor) || (T <: Real && t isa DiagonalTensorMap)
            end
        end

        @testset "Isometric projections" begin
            for T in eltypes,
                    t in (
                        randn(T, W, W), randn(T, W, W)',
                        randn(T, W, V4), randn(T, V4, W)',
                    )
                t2 = project_isometric(t)
                @test isisometric(t2)
                t3 = project_isometric(t2)
                @test t3 ≈ t2 # stability of the projection
                @test t2 * (t2' * t) ≈ t

                tc = similar(t)
                t3 = @constinferred project_isometric!(copy!(tc, t), t2)
                @test t3 === t2
                @test isisometric(t2)

                # test that t2 is closer to A then any other isometry
                for k in 1:10
                    δt = randn!(similar(t))
                    t3 = project_isometric(t + δt / 100)
                    @test norm(t - t3) > norm(t - t2)
                end
            end
        end
    end
end