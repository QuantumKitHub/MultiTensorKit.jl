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

blocksectors(sp ⊗ sp)
@plansor fullcont[] := A[a b;a b] # problem here is that fusiontrees for all 12 units are given

# 𝒞 x ℳ example
obj = A4Object(1,1,1)
obj2 = A4Object(1,2,1)

sp = Vect[A4Object](obj=>1)
sp2 = Vect[A4Object](obj2=>1)
@test_throws ArgumentError("invalid fusion channel") TensorMap(rand, ComplexF64, sp ⊗ sp2 ← sp)
homspace = sp ⊗ sp2 ← sp2
A = TensorMap(ones, ComplexF64, homspace)
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

blocksectors(sp ⊗ sp2)
A = TensorMap(ones, ComplexF64, sp ⊗ sp2 ← sp ⊗ sp2)
@plansor fullcont[] := A[a b;a b] # same 12 fusiontrees problem

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

@plansor Acont[] := A[a b;a b] # ignore this error for now
@plansor Acont2[] := A[b a;b a]

testsp = SU2Space(0=>1, 1=>1)
Atest = TensorMap(ones, ComplexF64, testsp ⊗ testsp ← testsp ⊗ testsp)
@plansor Aconttest[] := Atest[a b;a b]


# 𝒞 x ℳ ← ℳ x 𝒟
c = A4Object(1,1,1)
m = A4Object(1,2,1)
d = A4Object(2,2,1)
W = Vect[A4Object](c=>1) ⊗ Vect[A4Object](m=>1) ← Vect[A4Object](m=>1) ⊗ Vect[A4Object](d=>1)

# bram stuff

for i in 1:12, j in 1:12
    for a in A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i,j])
        F = Fsymbol(a, dual(a), a, a, leftone(a), rightone(a))[1,1,1,1]
        isapprox(F, frobeniusschur(a) / dim(a); atol=1e-15) || @show a, F, frobeniusschur(a)/ dim(a) # check real
        isreal(frobeniusschur(a)) || @show a, frobeniusschur(a)
    end
end

for i in 1:12, j in 1:12 # 18a
    i != j || continue
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i,j])
    @assert all(dim(m) > 0 for m in objsij)
end

for i in 1:12, j in 1:12 # 18b
    objsii = A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i,i])
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i,j])

    Ndict = Dict{Tuple{A4Object, A4Object, A4Object}, Int}()
    for a in objsii, m in objsij
        for n in a⊗m
            Ndict[(a, m, n)] = Nsymbol(a, m, n)
        end
    end

    for a in objsii, m in objsij
        isapprox(dim(a)*dim(m), sum(Ndict[(a, m, n)]*dim(n) for n in a⊗m); atol=2e-9) || @show a, m
    end
end

for i in 1:12, j in 1:12 # 18c
    objsii = A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i,i])
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i,j])
    m_dimsum = sum(dim(m)^2 for m in objsij)
    c_dimsum = sum(dim(c)^2 for c in objsii)
    isapprox(m_dimsum, c_dimsum; atol=1e-8) || @show i, j, c_dimsum, m_dimsum
end

############ MPSKit wow ############
using MultiTensorKit
using TensorKit
using MPSKit, MPSKitModels
C1 = A4Object(1,1,1)
C0 = A4Object(1,1,4) # unit
M = A4Object(1,2,1)
D0 = A4Object(2,2,12) # unit
D1 = A4Object(2,2,1)
collect(D0 ⊗ D1)
collect(D1 ⊗ D1)

P = Vect[A4Object](D0 => 1, D1 => 1)
h = TensorMap(ones, ComplexF64, P ⊗ P ← P ⊗ P)

lattice = InfiniteChain(1)
H = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(lattice));

D = 2
V = Vect[A4Object](M => D);
inf_init = InfiniteMPS([P], [V])

# (a, b, c, d, e, f) = (A4Object(2, 1, 1), A4Object(1, 2, 1), A4Object(2, 2, 11), A4Object(2, 2, 11), A4Object(2, 2, 9), A4Object(1, 2, 1))
# Fsymbol(a,b,c,d,e,f)
# zeros(ComplexF64, Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d))

# VUMPS
ψ, envs = find_groundstate(inf_init, H, VUMPS(verbosity=3, tol=1e-10, maxiter=500));
expectation_value(ψ, H, envs)

# testing blocksectors 
W = Vect[A4Object](A4Object(2, 2, 12)=>1) ← ProductSpace{GradedSpace{A4Object, NTuple{486, Int64}}, 0}() 
W isa TensorMapSpace{GradedSpace{A4Object, NTuple{486, Int64}}}
W isa HomSpace
W isa HomSpace{S} where {S<:SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}}
W isa TensorMapSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}}

# this appears as well (N1=N2=0)
W = ProductSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}, 0}() ← ProductSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}, 0}()
typeof(W)
W isa TensorMapSpace{S} where {S<:GradedSpace{A4Object, NTuple{486, Int64}}}
W isa HomSpace{S} where {S<:GradedSpace{A4Object, NTuple{486, Int64}}}
W isa HomSpace
W isa TensorMapSpace

W isa TensorMapSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}, 0, 0}
W isa TensorMapSpace{GradedSpace{A4Object, NTuple{486, Int64}}, 0, 0}
TensorMapSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}, 0, 0} <: TensorMapSpace{GradedSpace{A4Object, NTuple{486, Int64}}, N₁,N₂} where {N₁,N₂}
W isa TensorSpace{SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}}

GradedSpace{A4Object, NTuple{486, Int64}} <: BlockTensorKit.SumSpace{GradedSpace{A4Object, NTuple{486, Int64}}}

entropy(ψ)
entanglement_spectrum(ψ)
transfer_spectrum(ψ,sector=C0)
correlation_length(ψ,sector=C0)
norm(ψ)

#IDMRG

ψ, envs = find_groundstate(inf_init, H, IDMRG(verbosity=3, tol=1e-8, maxiter=100));
expectation_value(ψ, H, envs)

inf_init2 = InfiniteMPS([P,P], [V,V])
H2 = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(InfiniteChain(2)));
ψ2, envs2 = find_groundstate(inf_init2, H2, IDMRG2(verbosity=3, tol=1e-8, maxiter=100));
expectation_value(ψ2, H2, envs2)

#QuasiParticleAnsatz

momenta = range(0, 2π, 4)
excE, excqp = excitations(H, QuasiparticleAnsatz(ishermitian=false), momenta, ψ, envs, sector=C0, num=1);

# problem in QPA
# f1 = FusionTree{A4Object}((A4Object(1, 2, 1), A4Object(2, 1, 1), A4Object(1, 2, 1)), A4Object(1, 2, 1), (false, false, false), (A4Object(1, 1, 2),), (3, 2))
# f2 = FusionTree{A4Object}((A4Object(2, 1, 1), A4Object(1, 1, 4)), A4Object(2, 1, 1), (true, true), (), (1,))
# i = 2
# a = A4Object(1, 2, 1) # (f1.uncoupled[1], f1.innerlines..., f1.coupled)[i-1]
# b = A4Object(2, 1, 1) # f2.uncoupled[1]
# c = A4Object(1, 1, 4) # f2.uncoupled[2]
# d = A4Object(1, 1, 2) # (f1.uncoupled[1], f1.innerlines..., f1.coupled)[i]
# e = A4Object(1, 1, 1) # in a⊗b
# ep = A4Object(2, 1, 1) # f2.uncoupled[i]

# Fs = MultiTensorKit._get_Fcache(A4Object)
# i,j,k,l = 1,2,1,1
# colordict = Fs[i][i,j,k,l]
# colordict[(1,1,4,2,1,1)]
# Fsymbol(a,b,c,d,e,ep)
# s1, s2, s3, s4 = Nsymbol(a,b,e), Nsymbol(e,c,d), Nsymbol(b,c,ep), Nsymbol(a,ep,d)
# size = [s1, s2, s3, s4]

# using BenchmarkTools
# @btime for i in 1:4
#     size[i] == 0 ? size[i] = 1 : nothing
# end
# @btime for i in findall(iszero, size)
#     size[i] = 1
# end

# size
# zeros(sectorscalartype(A4Object), size...)

# util = similar(ψ.AL[1], space(parent(H)[1],1)[1])
# MPSKit.fill_data!(util, one)
# finite stuff
L = 6
# lattice = FiniteChain(L)
# P = Vect[A4Object](D0 => 1, D1 => 1)
# D = 2
# V = Vect[A4Object](M => D)

# fin_init = FiniteMPS(L, P, V, left=V, right=V)
# Hfin = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(lattice));
# ψ, envs = find_groundstate(fin_init, Hfin, DMRG(verbosity=3, tol=1e-8, maxiter=100));
# expectation_value(ψ, Hfin, envs)

# entropy(ψ, round(Int, L/2))
# entanglement_spectrum(ψ)
# exact_diagonalization(Hfin; sector=D0)

# ψ, envs = find_groundstate(fin_init, Hfin, DMRG2(verbosity=3, tol=1e-8, maxiter=100));
# expectation_value(ψ, Hfin, envs)

# entropy(ψ, round(Int, L/2))
# entanglement_spectrum(ψ)

# HfinPBC = periodic_boundary_conditions(Hfin,L);
# ψ, envs = find_groundstate(fin_init, HfinPBC, DMRG(verbosity=3, tol=1e-8, maxiter=100));
# expectation_value(ψ, HfinPBC, envs)

# # excitations
# excE, excqp = excitations(Hfin, QuasiparticleAnsatz(ishermitian=false), ψ, envs;sector=C0, num=1)