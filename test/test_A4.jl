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
end

for i in 1:r, j in 1:r
    @testset "Unitarity of $Istr F-move ($i, $j)" begin
        if i == j
            @testset "Unitarity of fusion F-move ($i, $j)" begin
                fusion_objects = I.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])
                @test unitarity_test(fusion_objects, fusion_objects, fusion_objects)
            end
        end

        i != j || continue # do this part only when off-diagonal
        mod_objects = I.(i, j, MultiTensorKit._get_dual_cache(I)[2][i, j])
        left_fusion_objects = I.(i, i, MultiTensorKit._get_dual_cache(I)[2][i, i])
        right_fusion_objects = I.(j, j, MultiTensorKit._get_dual_cache(I)[2][j, j])

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
            modop_objects = I.(j, i, MultiTensorKit._get_dual_cache(I)[2][j, i])

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
