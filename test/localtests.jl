using TensorKitSectors
using MultiTensorKit
using Revise

testobj = A4Object(1,1,1) # fusion cat object
unit = one(testobj)
collect(testobj⊗unit)
@assert unit == leftone(testobj) == rightone(testobj)

testobj2 = A4Object(2,2,1)
unit2 = one(testobj2)
collect(testobj2⊗unit2)
@assert unit2 == leftone(testobj2) == rightone(testobj2)

testmodobj = A4Object(1,2,1)
one(testmodobj)
leftone(testmodobj)
rightone(testmodobj)

Fsymbol(testobj, testobj, A4Object(1,1,3), testobj, A4Object(1,1,3), A4Object(1,1,4))

using Artifacts
using DelimitedFiles

artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.1")
filename = joinpath(artifact_path, "A4", "Fsymbol_4.txt")
txt_string = read(filename, String)
F_arraypart = copy(readdlm(IOBuffer(txt_string)));

MTK = MultiTensorKit
F_arraypart = MTK.convert_Fs(F_arraypart)
# for (color, colordict) in F_arraypart
#     for (labels, F) in colordict
#         if length(F) == 2
#             println(color, labels, F)
#         end
#     end
# end
i,j,k,l = (4,12,12,2) # 5,2,8,10
a,b,c,d,e,f = (1, 11, 3, 1, 1, 3) #(2,1,1,2,2,3)
testF = F_arraypart[(i,j,k,l)][(a,b,c,d,e,f)]
a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
                                                (k, l, c), (i, l, d),
                                                (i, k, e), (j, l, f)))
result = Array{ComplexF64,4}(undef,
                            (Nsymbol(a_ob, b_ob, e_ob),
                            Nsymbol(e_ob, c_ob, d_ob),
                            Nsymbol(b_ob, c_ob, f_ob),
                            Nsymbol(a_ob, f_ob, d_ob)))

map!(result, reshape(testF, size(result))) do pair
    return pair[2]
end

for c in testF[2]
    println(c)
end

function bla()
    return for (k,v) in F_arraypart
        @show k
    end
end

N = MultiTensorKit._get_Ncache(A4Object);

duals = MultiTensorKit._get_dual_cache(A4Object)[2]
# checking duals is correct
for i in 1:12, j in 1:12
    for (index, a) in enumerate(duals[i,j])
        aob = A4Object(i,j,index)
        bob = A4Object(j,i,a)
        leftone(aob) ∈ aob ⊗ bob && rightone(aob) ∈ bob ⊗ aob || @show i,j,aob,bob
    end
end

A = MultiTensorKit._get_Fcache(A4Object)
a,b,c,d,e,f = A4Object(1,1,3), A4Object(1,1,2), A4Object(1,1,2), A4Object(1,1,2), A4Object(1,1,2), A4Object(1,1,2)
a,b,c,d,e,f = A4Object(1,1,1), A4Object(1,1,2), A4Object(1,1,2), A4Object(1,1,1), A4Object(1,1,2), A4Object(1,1,4)
coldict = A[a.i][a.i, a.j, b.j, c.j]
bla = get(coldict, (a.label, b.label, c.label, d.label, e.label, f.label)) do
        return coldict[(a.label, b.label, c.label, d.label, e.label, f.label)]
    end

###### TensorKit stuff ######
using MultiTensorKit
using TensorKit
using Test

# 𝒞 x 𝒞 example

obj = A4Object(2,2,1)
obj2 = A4Object(2,2,2)
sp = Vect[A4Object](obj=>1, obj2=>1)
A = TensorMap(ones, ComplexF64, sp ⊗ sp ← sp ⊗ sp)
transpose(A, (2,4,), (1,3,))

# 𝒞 x ℳ example
obj = A4Object(1,1,1)
obj2 = A4Object(1,2,1)

sp = Vect[A4Object](obj=>1)
sp2 = Vect[A4Object](obj2=>1)
@test_throws ArgumentError("invalid fusion channel") TensorMap(rand, ComplexF64, sp ⊗ sp2 ← sp)
homspace = sp ⊗ sp2 ← sp2
A = TensorMap(ones, ComplexF64, homspace)
for sector in sectors(sp2)
    @show sector
end
fusiontrees(A)
permute(space(A),((1,),(3,2)))
transpose(A, (1,2,), (3,)) == A 
transpose(A, (3,1,), (2,))

Aop = TensorMap(ones, ComplexF64, conj(sp2) ⊗ sp ← conj(sp2))
transpose(Aop, (1,2,), (3,)) == Aop
transpose(Aop, (1,), (3,2))

@plansor Acont[a] := A[a b;b] # should not have data bc sp isn't the unit 

spfix = Vect[A4Object](one(obj)=>1)
Afix = TensorMap(ones, ComplexF64, spfix ⊗ sp2 ← sp2)
@plansor Acontfix[a] := Afix[a b;b] # should have a fusion tree

# completely off-diagonal example

obj = A4Object(5, 4, 1)
obj2 = A4Object(4, 5, 1)
sp = Vect[A4Object](obj=>1)
sp2 = Vect[A4Object](obj2=>1)
conj(sp) == sp2 

A = TensorMap(ones, ComplexF64, sp ⊗ sp2 ← sp ⊗ sp2)
Aop = TensorMap(ones, ComplexF64, sp2 ⊗ sp ← sp2 ⊗ sp)

At = transpose(A, (2,4,), (1,3,))
Aopt = transpose(Aop, (2,4,), (1,3,))

blocksectors(At) == blocksectors(Aop)
blocksectors(Aopt) == blocksectors(A)

@plansor Acont[] := A[a b;a b]