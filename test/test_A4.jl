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

i = 1
j = 2

V = (Vect[I]((i, i, label) => 1 for label in MTK._numlabels(I, i, i)),
        Vect[I]((i, i, 1) => 1, (i, i, 2) => 2),
        Vect[I]((i, i, 1) => 1, (i, i, 2) => 1),
        Vect[I]((i, i, label) => 1 for label in MTK._numlabels(I, i, i)),
        Vect[I]((i, i, 1) => 2, (i, i, 3) => 3))

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

    # CONTINUE HERE

    @timedtestset "HomSpace with $(TK.type_repr(Vect[I])) " begin
        for (V1, V2, V3, V4, V5) in (VIBC, VIBD, VIBM1, VIBM2, VIBMop1, VIBMop2)
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

@testset "$Istr ($i, $j) left and right units" for i in 1:r, j in 1:r
    Cij_obs = I.(i, j, MTK._get_dual_cache(I)[2][i, j])

    s = rand(Cij_obs)
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
