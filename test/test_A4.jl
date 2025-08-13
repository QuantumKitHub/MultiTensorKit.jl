using MultiTensorKit
using TensorKitSectors, TensorKit
using Test, TestExtras
using Random
using LinearAlgebra: LinearAlgebra

const MTK = MultiTensorKit
const TK = TensorKit

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
    @test_throws ArgumentError one(I)
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
            @test isone(@constinferred(one(s[1])))
            u = I.(i, i, MTK._get_dual_cache(I)[1][i])
            @test u == @constinferred(leftone(u)) == @constinferred(rightone(u)) ==
                @constinferred(one(u))
            @test isone(@constinferred(one(s[1])))
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

            @test (isone(@constinferred(leftone(m))) && isone(@constinferred(rightone(m))))
            @test one(c) == leftone(m) == rightone(mop)
            @test one(d) == rightone(m) == leftone(mop)
            @test_throws DomainError one(m)
            @test_throws DomainError one(mop)

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

            @test Nsymbol(c, one(c), c) == Nsymbol(d, one(d), d) == 1

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
        @test eval(Meta.parse(TK.type_repr(typeof(V)))) == typeof(V)
        @test eval(Meta.parse(sprint(show, V))) == V
        @test eval(Meta.parse(sprint(show, V'))) == V'
        @test V' == GradedSpace(gen; dual=true)
        @test V == @constinferred GradedSpace(gen...)
        @test V' == @constinferred GradedSpace(gen...; dual=true)
        @test V == @constinferred GradedSpace(tuple(gen...))
        @test V' == @constinferred GradedSpace(tuple(gen...); dual=true)
        @test V == @constinferred GradedSpace(Dict(gen))
        @test V' == @constinferred GradedSpace(Dict(gen); dual=true)
        @test V == @inferred Vect[I](gen)
        @test V' == @constinferred Vect[I](gen; dual=true)
        @test V == @constinferred Vect[I](gen...)
        @test V' == @constinferred Vect[I](gen...; dual=true)
        @test V == @constinferred Vect[I](Dict(gen))
        @test V' == @constinferred Vect[I](Dict(gen); dual=true)
        @test V == @constinferred typeof(V)(c => dim(V, c) for c in sectors(V))
        @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
        @test V == GradedSpace(reverse(collect(gen))...)
        @test eval(Meta.parse(sprint(show, V))) == V
        @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)

        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test isa(InnerProductStyle(V), HasInnerProduct)
        @test isa(InnerProductStyle(V), EuclideanInnerProduct)
        @test isa(V, GradedSpace)
        @test isa(V, GradedSpace{I})
        @test @constinferred(dual(V)) == @constinferred(conj(V)) ==
              @constinferred(adjoint(V)) != V
        @test @constinferred(field(V)) == ℂ
        @test @constinferred(sectortype(V)) == I
        slist = @constinferred sectors(V)
        @test @constinferred(hassector(V, first(slist)))
        @test @constinferred(dim(V)) == sum(dim(s) * dim(V, s) for s in slist)
        @test @constinferred(reduceddim(V)) == sum(dim(V, s) for s in slist)
        @constinferred dim(V, first(slist))

        @test @constinferred(⊕(V, zero(V))) == V
        @test @constinferred(⊕(V, V)) == Vect[I](c => 2dim(V, c) for c in sectors(V))
        @test @constinferred(⊕(V, V, V, V)) == Vect[I](c => 4dim(V, c) for c in sectors(V))

        @testset "$Istr ($i, $j) spaces" for i in 1:r, j in 1:r
            # space with a single sector
            Wleft = @constinferred Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i))
            Wright = @constinferred Vect[I]((j, j, label) => 1 for label in 1:MTK._numlabels(I, j, j))
            WM = @constinferred Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j))
            WMop = @constinferred Vect[I]((j, i, label) => 1 for label in 1:MTK._numlabels(I, j, i))

            @test @constinferred(oneunit(Wleft)) == leftoneunit(Wleft) == rightoneunit(Wleft)
            @test @constinferred(oneunit(Wright)) == leftoneunit(Wright) == rightoneunit(Wright)
            @test @constinferred(leftoneunit(⊕(Wleft, WM))) == oneunit(Wleft)
            @test @constinferred(leftoneunit(⊕(Wright, WMop))) == oneunit(Wright)
            @test @constinferred(rightoneunit(⊕(Wright, WM))) == oneunit(Wright)
            @test @constinferred(rightoneunit(⊕(Wleft, WMop))) == oneunit(Wleft)

            @test_throws ArgumentError oneunit(I)

            if i != j # some tests specialised for modules
                @test_throws ArgumentError oneunit(WM)
                @test_throws ArgumentError oneunit(WMop)

                # sensible direct sums and fuses
                ul, ur = one(I(i, i, 1)), one(I(j, j, 1))
                @test @constinferred(⊕(Wleft, WM)) ==
                    Vect[I](c => 1 for c in sectors(V) if leftone(c) == ul == rightone(c) || (c.i == i && c.j == j))
                @test @constinferred(⊕(Wright, WMop)) ==
                    Vect[I](c => 1 for c in sectors(V) if leftone(c) == ur == rightone(c) || (c.i == j && c.j == i))
                @test @constinferred(⊕(Wright, WM)) ==
                    Vect[I](c => 1 for c in sectors(V) if rightone(c) == ur == leftone(c) || (c.i == i && c.j == j))
                @test @constinferred(⊕(Wleft, WMop)) ==
                    Vect[I](c => 1 for c in sectors(V) if rightone(c) == ul == leftone(c) || (c.i == j && c.j == i))
                @test @constinferred(fuse(Wleft, WM)) == Vect[I](c => dim(Wleft) for c in sectors(WM)) # this might be wrong
                @test @constinferred(fuse(Wright, WMop)) == Vect[I](c => dim(Wright) for c in sectors(WMop)) # same

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
                @test @constinferred(⊕(W, oneunit(W))) ==
                    Vect[I](c => isone(c) + dim(W, c) for c in sectors(W))
                @test @constinferred(fuse(W, oneunit(W))) == W
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
        @test infimum(V, GradedSpace(s => randlen)) ==
              GradedSpace(s => randlen)
        @test_throws SpaceMismatch (⊕(V, V'))
    end

    @timedtestset "HomSpace with $(TK.type_repr(Vect[I])) involving ($i, $j)" for i in 1:r, j in 1:r
        V = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
        Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)),
        Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
        Vect[I]((i, j, 1) => 3),
        Vect[I]((j, j, label) => 1 for label in 1:MTK._numlabels(I, j, j)))

        for (V1, V2, V3, V4, V5) in (V,)
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

            @test_throws ErrorException insertleftunit(W)
            @test insertrightunit(W) == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ rightoneunit(V5))
            @test_throws ErrorException insertrightunit(W, 6)
            @test_throws ErrorException insertleftunit(W, 6)

            @test (V1 ⊗ V2 ⊗ rightoneunit(V2) ← V3 ⊗ V4 ⊗ V5) ==
                  @constinferred(insertrightunit(W, 2))
            @test (V1 ⊗ V2 ← leftoneunit(V3) ⊗ V3 ⊗ V4 ⊗ V5) ==
                  @constinferred(insertleftunit(W, 3))
            @test @constinferred(removeunit(insertleftunit(W, 3), 3)) == W
            @test_throws ErrorException @constinferred(insertrightunit(one(V1) ← V1, 0)) # should I specify it's the other error?
            @test_throws ErrorException insertleftunit(one(V1) ← V1, 0)
        end
    end
end

println("---------------------------------------")
println("|    Multifusion fusion tree tests    |")
println("---------------------------------------")

@timedtestset "Fusion trees for $(TK.type_repr(I)) involving ($i, $j)" verbose = true for i in 1:r, j in 1:r
    N = 6
    Mop = rand_object(I, j, i)
    M = rand_object(I, i, j)
    C0 = one(I(i, i, 1))
    C1 = rand_object(I, i, i) 
    D0 = one(I(j, j, 1))
    D1 = rand_object(I, j, j)
    out = (Mop, C0, C1, M, D0, D1) # should I try to make a non-hardcoded example? could vary number of Cs and Ds, as well as randomly fuse and check if allowed
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

    @testset "Fusion tree $Istr: constructor properties" for u in (C0, D0)
        @constinferred FusionTree((), u, (), (), ())
        @constinferred FusionTree((u,), u, (false,), (), ())
        @constinferred FusionTree((u, u), u, (false, false), (), (1,))
        @constinferred FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @constinferred FusionTree((u, u, u, u), u, (false, false, false, false), (u, u),
                                  (1, 1, 1))
        @test_throws MethodError FusionTree((u, u, u), u, (false, false), (u,), (1, 1))
        @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (u, u),
                                            (1, 1))
        @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (u,),
                                            (1, 1, 1))
        @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (), (1,))

        f = FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @test sectortype(f) == I
        @test length(f) == 3
        @test FusionStyle(f) == FusionStyle(I)
        @test BraidingStyle(f) == BraidingStyle(I)

        # SimpleFusion
        errstr = "fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`"
        @test_throws errstr FusionTree((), u, ())
        @test_throws errstr FusionTree((u,), u, (false,))
        @test_throws errstr FusionTree((u, u), u, (false, false))
        @test_throws errstr FusionTree((u, u, u), u)
        @test_throws errstr FusionTree((u, u, u, u)) # custom FusionTree constructor required here
    end

    @testset "Fusion tree $Istr: insertat" begin
        N = 4
        out2 = random_fusion(I, i, j, N)
        in2 = rand(collect(⊗(out2...)))
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for k in 1:N
            out1, in1 = nothing, nothing
            while in1 === nothing
                try
                    out1 = random_fusion(I, i, j, N) # guaranteed good fusion
                    out1 = Base.setindex(out1, in2, k) # can lead to poor fusion
                    in1 = rand(collect(⊗(out1...)))
                catch e
                    if isa(e, AssertionError)
                        in1 = nothing # keep trying till out1 is compatible with inserting in2 at k
                    else
                        rethrow(e)
                    end
                end
            end
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
        tp = safe_tensor_product(in1, in2) # messy solution but it works
        while tp === nothing
            out1 = random_fusion(I, i, j, N)
            out2 = random_fusion(I, i, j, N)
            in1 = rand(collect(⊗(out1...)))
            in2 = rand(collect(⊗(out2...)))
            tp = safe_tensor_product(in1, in2)
        end

        f1 = rand(collect(fusiontrees(out1, in1)))
        f2 = rand(collect(fusiontrees(out2, in2)))


        @test dim(in1) * dim(in2) ≈ sum(abs2(coeff) * dim(c) for c in in1 ⊗ in2
                                        for μ in 1:Nsymbol(in1, in2, c)
                                        for (f, coeff) in TK.merge(f1, f2, c, μ))
        # no merge and braid interplay tests
    end

    # hardcoded double fusion tree tests
    N = 6
    out = (Mop, C0, C1, M, D0, D1) # same as above
    out2 = (D0, D1, Mop, C0, C1, M) # different order that still fuses to D0 or D1

    incoming = rand(collect(⊗(out...))) # will be in 𝒞ⱼⱼ
    while incoming ∉ collect(⊗(out2...)) # when i = j these don't necessarily fuse to the same object, since Mop x M doesn't return all objects in 𝒞ᵢᵢ
        Mop = rand_object(I, j, i)
        out2 = (D0, D1, Mop, C0, C1, M)
    end

    f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
    f2 = rand(collect(fusiontrees(out2, incoming, ntuple(n -> rand(Bool), N))))

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
            @test eltype(bs) === Pair{typeof(c),typeof(b1)}
            @test typeof(b1) === TK.blocktype(t)
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
            @test norm(one!(t) - id(V)) == 0

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
            @test convert(TensorMap, t1 + t2) ≈
                  convert(TensorMap, t1) + convert(TensorMap, t2)
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
        @test all(Base.Fix2(isa, DiagonalTensorMap),
                  (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1)))
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
    @timedtestset "Factorization" begin
        for T in (Float32, ComplexF64)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @testset "eig" begin
                D, W = @constinferred eig(t)
                @test t * W ≈ W * D
                t2 = t + t'
                D2, V2 = @constinferred eigh(t2)
                VdV2 = V2' * V2
                @test VdV2 ≈ one(VdV2)
                @test t2 * V2 ≈ V2 * D2

                @test rank(D) ≈ rank(t)
                @test cond(D) ≈ cond(t)
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.eigvals(D)),
                              values(LinearAlgebra.eigvals(t))))
            end
            @testset "leftorth with $alg" for alg in (TK.QR(), TK.QL())
                Q, R = @constinferred leftorth(t; alg=alg)
                QdQ = Q' * Q
                @test QdQ ≈ one(QdQ)
                @test Q * R ≈ t
                if alg isa Polar
                    @test isposdef(R)
                end
            end
            @testset "rightorth with $alg" for alg in (TK.RQ(), TK.LQ())
                L, Q = @constinferred rightorth(t; alg=alg)
                QQd = Q * Q'
                @test QQd ≈ one(QQd)
                @test L * Q ≈ t
                if alg isa Polar
                    @test isposdef(L)
                end
            end
            @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                U, S, Vᴴ = @constinferred tsvd(t; alg=alg)
                UdU = U' * U
                @test UdU ≈ one(UdU)
                VdV = Vᴴ * Vᴴ'
                @test VdV ≈ one(VdV)
                @test U * S * Vᴴ ≈ t

                @test rank(S) ≈ rank(t)
                @test cond(S) ≈ cond(t)
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.svdvals(S)),
                              values(LinearAlgebra.svdvals(t))))
            end
        end
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

@timedtestset "Tensors with symmetry involving $Istr ($i, $j)" verbose = true for i in 1:r, j in 1:r
    VC = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
            Vect[I](one(I(i, i, 1)) => 2),
            Vect[I](one(I(i, i, 1)) => 2, rand_object(I, i, i) => 1),
            Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)),
            Vect[I](one(I(i, i, 1)) => 2, rand_object(I, i, i) => 3)
    )

    VM1 = (Vect[I]((i, i, label) => 1 for label in 1:MTK._numlabels(I, i, i)), # written so V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 works
            Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)), # important that V4 is module-graded
            Vect[I](one(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            Vect[I](rand_object(I, i, j) => 1),
            Vect[I](one(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    VM2 = (Vect[I]((i, j, label) => 1 for label in 1:MTK._numlabels(I, i, j)), # second set where module is V1 here
            Vect[I]((j, j, label) => 1 for label in 1:MTK._numlabels(I, j, j)),
            Vect[I](one(I(i, i, 1)) => 1, rand_object(I, i, i) => 1),
            Vect[I](rand_object(I, i, j) => 2),
            Vect[I](one(I(j, j, 1)) => 2, rand_object(I, j, j) => 1)
    )

    Vcol = i != j ? (VM1, VM2) : (VC,) # avoid duplicate runs

    for V in Vcol
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = i == j ? V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5 : V3 ⊗ V4 ⊗ V5 # fusion matters
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
                t = @constinferred zeros(T, W) # empty for i != j b/c blocks are module-graded
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T,spacetype(t),length(W),0,Vector{T}}
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
                @test eltype(bs) === Pair{typeof(c),typeof(b1)}
                @test typeof(b1) === TK.blocktype(t')
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

                if all(a.i == a.j for a in blocksectors(W)) # can't reverse fusion for these
                    i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1))
                    i2 = @constinferred(isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2))
                    @test i1 * i2 == @constinferred(id(T, V1 ⊗ V2))
                    @test i2 * i1 == @constinferred(id(Vector{T}, V2 ⊗ V1))
                end
                for v in (V1, V2, V3, V4, V5)
                    wl = @constinferred(isometry(T, (leftoneunit(v) ⊕ leftoneunit(v)) ⊗ v, v))
                    wr = @constinferred(isometry(T, v ⊗ (rightoneunit(v) ⊕ rightoneunit(v)), v))
                    for w in (wl, wr)
                        @test dim(w) == 2 * dim(v ← v)
                        @test w' * w == id(Vector{T}, v)
                        @test w * w' == (w * w')^2
                    end
                end
            end
        end

        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W) # fusion matters here
                t2 = @constinferred insertleftunit(t, 5) # default errors

                @test t2 == @constinferred insertrightunit(t, 4) # default doesn't error bc i==N then
                @test numind(t2) == numind(t) + 1
                @test space(t2) == insertleftunit(space(t), 5)
                @test scalartype(t2) === T
                @test t.data === t2.data
                @test @constinferred(removeunit(t2, $(numind(t2) - 1))) == t # -1 required

                t3 = @constinferred insertleftunit(t, 5; copy=true) # same here
                @test t3 == @constinferred insertrightunit(t, 4; copy=true)
                @test t.data !== t3.data
                for (c, b) in blocks(t)
                    @test b == block(t3, c)
                end
                @test @constinferred(removeunit(t3, $(numind(t3) - 1))) == t
                t4 = @constinferred insertrightunit(t, 3; dual=true)
                @test numin(t4) == numin(t) + 1 && numout(t4) == numout(t)
                for (c, b) in blocks(t)
                    @test b == block(t4, c)
                end
                @test @constinferred(removeunit(t4, 4)) == t
                t5 = @constinferred insertleftunit(t, 4; dual=true)
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
            t = rand(ComplexF64, V1 ⊗ V2 ← V1 ⊗ V2)
            s = @constinferred tr(t)
            @test conj(s) ≈ tr(t')
            try # needed for module cases: certain transposes with module legs will result in different colorings
                @planar s2 = t[a b; a b] # no twist needed bc permute avoided
                @test s ≈ s2
            catch e
                @test isa(e, SectorMismatch)
            end

            try # TODO?: skip module traces
                @planar t3[a; b] := t[a c; b c]
                @planar s3 = t3[a; a] # this contraction order gives zero for VIBMop1 and VIBMop2 because it traces out the module legs
                @test s ≈ s3
            catch e
                @test isa(e, SectorMismatch)
            end
        end

        @timedtestset "Partial trace: test self-consistency" begin
            t = rand(ComplexF64, V3 ⊗ V4 ⊗ V5 ← V3 ⊗ V4 ⊗ V5) # rewritten to be compatible with module fusion
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
            W1 = W2 ⊗ (rightoneunit(V5) ⊕ rightoneunit(V5))
            W3 = (leftoneunit(V4) ⊕ leftoneunit(V4)) ⊗ W2
            for W in (W1, W3)
                for T in (Float64, ComplexF64)
                    t1 = @constinferred randisometry(T, W, W2)
                    t2 = randisometry(T, W2 ← W2)
                    @test t1' * t1 ≈ one(t2)
                    @test t2' * t2 ≈ one(t2)
                    @test t2 * t2' ≈ one(t2)
                    P = t1 * t1'
                    @test P * P ≈ P
                end
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
                # @test_throws SpaceMismatch inv(t) # can coincidently fail b/c of rand_object
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

        # some fail for (2, 2), (3, 3), (6, 6)
        # rightorth RQ(pos) and Polar (fail) for 2nd space
        # leftorth with QL(pos) and Polar for 1st space
        # leftnull QR for 1st space
        # cond and rank leftnull for 1st space
        @timedtestset "Factorization" begin
            WL = V3 ⊗ V4 ⊗ V2 ← V1' ⊗ V5' # old left permute resulted in this space
            WR = V3 ⊗ V4 ← V2' ⊗ V1' ⊗ V5' # old right permute
            WmodR = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 # new fusion order for right
            WmodL = V1 ⊗ V2 ⊗ V5' ← V3 ⊗ V4 # new fusion order for left

            isdiag = all(c.i == c.j for c in blocksectors(WmodR)) # this blocksectors call should always work
            for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                # adjoint takes other space for shape of matrix in RQ(pos)
                tsR = isdiag ? (rand(T, WR), rand(T, WL)') : (rand(T, WmodR), rand(T, WmodL)') # shape of matrices require different spaces for left/right
                tsL = isdiag ? (rand(T, WL), rand(T, WR)') : (rand(T, WmodR), rand(T, WmodL)')
                for t in tsR
                    @testset "rightorth with $alg" for alg in
                                                    (TK.RQ(), TK.RQpos(), TK.LQ(),
                                                        TK.LQpos(),
                                                        TK.Polar(), TK.SVD(), TK.SDD())
                        (alg isa RQ || alg isa RQpos || alg isa Polar) && !isdiag && continue
                        L, Q = @constinferred rightorth(t; alg=alg)
                        QQd = Q * Q'
                        @test QQd ≈ one(QQd)
                        @test L * Q ≈ t
                        if alg isa Polar
                            @test isposdef(L)
                            @test domain(L) == codomain(L) == space(t, 1) ⊗ space(t, 2)
                        end
                    end
                    @testset "rightnull with $alg" for alg in (TK.LQ(), TK.SVD(), TK.SDD())
                        M = @constinferred rightnull(t; alg=alg)
                        MMd = M * M'
                        @test MMd ≈ one(MMd)
                        @test norm(t * M') < 100 * eps(norm(t))
                    end
                end
                # adjoints take other space for shape of matrix in QL(pos)
                for t in tsL
                    @testset "leftorth with $alg" for alg in
                                                    (TK.QR(), TK.QRpos(), TK.QL(), TK.QLpos(),
                                                    TK.Polar(), TK.SVD(), TK.SDD())
                        # skip QL because the monomorphism condition is hard to satisfy for off-diagonal case
                        # have to skip Polar as well as all tests fail with modules
                        (alg isa QL || alg isa QLpos || alg isa Polar) && !isdiag && continue
                        Q, R = @constinferred leftorth(t; alg=alg)
                        QdQ = Q' * Q
                        @test QdQ ≈ one(QdQ)
                        @test Q * R ≈ t
                        if alg isa Polar
                            @test isposdef(R) # this fails with modules
                            @test domain(R) == codomain(R) == space(t, 4)' ⊗ space(t, 5)' # this as well
                        end
                    end
                    @testset "leftnull with $alg" for alg in
                                                    (TK.QR(), TK.SVD(), TK.SDD())
                        # less rows than columns so either fails or no data in off-diagonal case
                        !isdiag && continue
                        N = @constinferred leftnull(t; alg=alg)
                        NdN = N' * N
                        @test NdN ≈ one(NdN)
                        @test norm(N' * t) < 100 * eps(norm(t))
                    end
                    @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                        U, S, V = @constinferred tsvd(t; alg=alg)
                        UdU = U' * U
                        @test UdU ≈ one(UdU)
                        VVd = V * V'
                        @test VVd ≈ one(VVd)
                        @test U * S * V ≈ t

                        s = LinearAlgebra.svdvals(t)
                        s′ = LinearAlgebra.diag(S)
                        for (c, b) in s
                            @test b ≈ s′[c]
                        end
                    end
                    @testset "cond and rank" begin
                        d1 = dim(codomain(t))
                        d2 = dim(domain(t))
                        # @test rank(t) ≈ min(d1, d2) # reduced to approx due to numerical F-symbols FIXME: fails sometimes for modules
                        if isdiag # leftnull doesn't work for off-diagonal case
                            M = leftnull(t)
                            @test rank(M) ≈ max(d1, d2) - min(d1, d2) # reduced to approx
                        end
                        t2 = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                        @test cond(t2) ≈ one(real(T))
                        @test rank(t2) ≈ dim(V1 ⊗ V2) # reduced to approx
                        t3 = randn(T, V1 ⊗ V2, V1 ⊗ V2)
                        t3 = (t3 + t3') / 2
                        vals = LinearAlgebra.eigvals(t3)
                        λmax = maximum(s -> maximum(abs, s), values(vals))
                        λmin = minimum(s -> minimum(abs, s), values(vals))
                        @test cond(t3) ≈ λmax / λmin
                    end
                end

                # how useful is this test? everything just works regardless of the space
                @testset "empty tensor" begin
                    t = randn(T, V1 ⊗ V2, zero(V1))
                    @testset "leftorth with $alg" for alg in
                                                    (TK.QR(), TK.QRpos(), TK.QL(), TK.QLpos(),
                                                    TK.Polar(), TK.SVD(), TK.SDD())
                        Q, R = @constinferred leftorth(t; alg=alg)
                        @test Q == t
                        @test dim(Q) == dim(R) == 0
                    end
                    @testset "leftnull with $alg" for alg in (TK.QR(), TK.SVD(), TK.SDD())
                        N = @constinferred leftnull(t; alg=alg)
                        @test N' * N ≈ id(domain(N))
                        @test N * N' ≈ id(codomain(N))
                    end
                    @testset "rightorth with $alg" for alg in
                                                    (TK.RQ(), TK.RQpos(), TK.LQ(),
                                                        TK.LQpos(),
                                                        TK.Polar(), TK.SVD(), TK.SDD())
                        L, Q = @constinferred rightorth(copy(t'); alg=alg)
                        @test Q == t'
                        @test dim(Q) == dim(L) == 0
                    end
                    @testset "rightnull with $alg" for alg in (TK.LQ(), TK.SVD(), TK.SDD())
                        M = @constinferred rightnull(copy(t'); alg=alg)
                        @test M * M' ≈ id(codomain(M))
                        @test M' * M ≈ id(domain(M))
                    end
                    @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                        U, S, V = @constinferred tsvd(t; alg=alg)
                        @test U == t
                        @test dim(U) == dim(S) == dim(V)
                    end
                    @testset "cond and rank" begin
                        @test rank(t) == 0
                        W2 = zero(V1) * zero(V2)
                        t2 = rand(W2, W2)
                        @test rank(t2) == 0
                        @test cond(t2) == 0.0
                    end
                end
                t = rand(T, V1 ⊗ V2 ← V1 ⊗ V2)
                @testset "eig and isposdef" begin
                    D, V = eigen(t)
                    @test t * V ≈ V * D

                    d = LinearAlgebra.eigvals(t; sortby=nothing)
                    d′ = LinearAlgebra.diag(D)
                    for (c, b) in d
                        @test b ≈ d′[c]
                    end

                    # Somehow moving these test before the previous one gives rise to errors
                    # with T=Float32 on x86 platforms. Is this an OpenBLAS issue? 
                    VdV = V' * V
                    VdV = (VdV + VdV') / 2
                    @test isposdef(VdV)

                    @test !isposdef(t) # unlikely for non-hermitian map
                    t2 = (t + t')
                    D, V = eigen(t2)
                    VdV = V' * V
                    @test VdV ≈ one(VdV)
                    D̃, Ṽ = @constinferred eigh(t2)
                    @test D ≈ D̃
                    @test V ≈ Ṽ
                    λ = minimum(minimum(real(LinearAlgebra.diag(b)))
                                for (c, b) in blocks(D))
                    @test cond(Ṽ) ≈ one(real(T))
                    @test isposdef(t2) == isposdef(λ)
                    @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                    @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
                end
            end
        end

        @timedtestset "Tensor truncation" begin
            for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                ts = (randn(T, V1 ⊗ V2, V3 ⊗ V4 ⊗ V5), randn(T, V4 ⊗ V5, V3 ⊗ V1 ⊗ V2)') # rewritten for modules
                for p in (1, 2, 3, Inf)
                    for t in ts
                        U₀, S₀, V₀ = tsvd(t)
                        t = rmul!(t, 1 / norm(S₀, p))
                        U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))),
                                            p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
                        U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    end
                end
            end
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
                # no reshape test: NoBraiding and no fusion tensor
            end
        end

        @timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                t1 = rand(T, V3 ⊗ V4 ⊗ V5 ← V1 ⊗ V2)
                if all(a.i != a.j for a in blocksectors(t1))
                    t2 = rand(T, V5' ⊗ V4' ⊗ V3', V2' ⊗ V1')
                else
                    t2 = rand(T, V3' ⊗ V1, V4 ⊗ V5 ⊗ V2') # keep a non-trivial permutation in diagonal case
                end
                t = @constinferred (t1 ⊗ t2)
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end

        @timedtestset "Tensor product: test via tensor contraction" begin
            # W = V3 ⊗ V4 ⊗ V5 ← V1 ⊗ V2
            W = V4 ← V1 ⊗ V2 # less costly
            isdiag = all(a.i == a.j for a in blocksectors(W))
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
end