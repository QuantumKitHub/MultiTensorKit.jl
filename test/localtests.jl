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

(a, b, c, d, e, f) = (A4Object(2, 1, 1), A4Object(1, 2, 1), A4Object(2, 2, 11), A4Object(2, 2, 11), A4Object(2, 2, 9), A4Object(1, 2, 1))
Fsymbol(a,b,c,d,e,f)
zeros(ComplexF64, Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d))

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

############ MPSKit wow ############
using MultiTensorKit
using TensorKit
using MPSKit, MPSKitModels
C1 = A4Object(1,1,1)
C0 = A4Object(1,1,4) # unit
M = A4Object(1,2,1)
D0 = A4Object(2,2,12) # unit
D1 = A4Object(2,2,2) # self-dual object
collect(D0 ⊗ D1)
collect(D1 ⊗ D1)

P = Vect[A4Object](D0 => 1, D1 => 1)
h = TensorMap(ones, ComplexF64, P ⊗ P ← P ⊗ P)

lattice = InfiniteChain(1);
H = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(lattice));

# testing insertleft/rightunit
sp = SU2Space(0=>1, 1=>1)
ht = TensorMap(ones, ComplexF64, P  ← P)
htl = TensorMap(ones, ComplexF64, P  ← one(P))
htr = TensorMap(ones, ComplexF64, one(P)  ← P)
htnone = TensorMap(ones, ComplexF64, one(P)  ← one(P))
insertrightunit(htr) # adding to empty space
insertleftunit(htr) 
insertrightunit(htl)
insertleftunit(htl) # adding to empty space
insertrightunit(htnone)
insertleftunit(htnone)


D = 2
V = Vect[A4Object](M => D);
inf_init = InfiniteMPS([P], [V]);

# VUMPS
ψ, envs = find_groundstate(inf_init, H, VUMPS(verbosity=3, tol=1e-10, maxiter=15));
expectation_value(ψ, H, envs)

entropy(ψ)
entanglement_spectrum(ψ)
transfer_spectrum(ψ,sector=C0)
correlation_length(ψ,sector=C0)
norm(ψ)

#IDMRG

ψ, envs = find_groundstate(inf_init, H, IDMRG(verbosity=3, tol=1e-8, maxiter=15));
expectation_value(ψ, H, envs)

#IDMRG2
inf_init2 = InfiniteMPS([P,P], [V,V])
H2 = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(InfiniteChain(2)));
idmrg2alg = IDMRG2(verbosity=3, tol=1e-8, maxiter=15, trscheme=truncdim(10))
ψ2, envs2 = find_groundstate(inf_init2, H2, idmrg2alg);
expectation_value(ψ2, H2, envs2)

#QuasiParticleAnsatz

momenta = range(0, 2π, 5)
excE, excqp = excitations(H, QuasiparticleAnsatz(ishermitian=false), momenta, ψ, envs, sector=C0, num=1); # not working for some reason

# quick test on complex f symbols and dimensions
testp = Vect[A4Object](one(A4Object(i,i,1)) => 1 for i in 1:12)
dim(testp)
oneunit(testp)

# finite stuff
L = 6
lattice = FiniteChain(L)
P = Vect[A4Object](D0 => 1, D1 => 1)
D = 2
V = Vect[A4Object](M => D)

dmrgalg = DMRG(verbosity=3, tol=1e-8, maxiter=15, alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false))
fin_init = FiniteMPS(L, P, V, left=V, right=V)
Hfin = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(lattice));
open_boundary_conditions(H, L) == Hfin
ψfin, envsfin = find_groundstate(fin_init, Hfin, dmrgalg);
expectation_value(ψfin, Hfin, envsfin) / (L-1)

entropy(ψfin, round(Int, L/2))
entanglement_spectrum(ψfin, round(Int, L/2))
Es, states, convhist = exact_diagonalization(Hfin; sector=D0);
Es / (L-1)

#DMRG2 weird real data incompatibility with sector type A4Object
dmrg2alg = DMRG2(verbosity=3, tol=1e-8, maxiter=15; alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false), trscheme=truncdim(10))
ψfin2, envsfin2 = find_groundstate(fin_init, Hfin, dmrg2alg);
expectation_value(ψfin2, Hfin, envsfin2) / (L-1)

entropy(ψfin2, round(Int, L/2))
entanglement_spectrum(ψfin2, round(Int, L/2))

S = left_virtualspace(Hfin, 1)
oneunit(S)
eltype(S)
oneunit(eltype(S)) # should error

# excitations
excEfin, excqpfin = excitations(Hfin, QuasiparticleAnsatz(ishermitian=false), ψfin, envsfin;sector=C0, num=1);
excEfin

excFIN, excqpFIN = excitations(Hfin, FiniteExcited(;gsalg=dmrg2alg), ψfin;num=1);
excFIN

# changebonds test
dim(left_virtualspace(ψ, 1))
ψch, envsch = changebonds(ψ, H, OptimalExpand(; trscheme=truncerr(1e-3)), envs)
dim(left_virtualspace(ψch, 1))

# time evolution

ψt, envst = timestep(ψ, H, 10, 0, TDVP(integrator=MPSKit.Defaults.alg_expsolve(; ishermitian=false)), envs); # not working for some reason
et = expectation_value(ψt, H, envst) 
e = expectation_value(ψ, H, envs)
isapprox(et, e*exp(-1im * 10 * e); atol=1e-1) # not hermitian

tdvpalg = TDVP(integrator=MPSKit.Defaults.alg_expsolve(; ishermitian=false, krylovdim=10))
ψt, envst = time_evolve(ψ, H, range(0, 1, 10), tdvpalg, envs)

make_time_mpo(H, 0.1, alg=WII(tol=1e-8, maxiter=100))

# stat-mech stuff
mpo = InfiniteMPO([h])
ψ, envs = leading_boundary(inf_init, mpo, VUMPS(verbosity=3, tol=1e-8, maxiter=15, alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));

# addition, substraction, multiplication

# finite 
Hfin * ψfin

# infinite
H * ψ # currently doesn't work
MPSKit.DenseMPO(H) * ψ # does work


# approximate
ψa, _ = approximate(ψ, (mpo, inf_init), IDMRG());

mpo2 = InfiniteMPO([h,h])
inf_init2 = InfiniteMPS([P,P], [V,V])
H2 = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(InfiniteChain(2)));
ψ2, envs2 = find_groundstate(inf_init2, H2, VUMPS(verbosity=3, tol=1e-10, maxiter=15));

ψa, _ = approximate(ψ2, (mpo2, inf_init2), IDMRG2(trscheme=truncdim(10)));

# # testing InfiniteMPOHamiltonian and FiniteMPOHamiltonian constructor not relying on MPSKitModels

sp = Vect[FibonacciAnyon](:I=>1, :τ => 1)
t = TensorMap(ones, ComplexF64, sp ← sp)
InfiniteMPOHamiltonian(PeriodicArray([sp]), i => t for i in 1:1)
H = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(InfiniteChain(1)));

# gradient grassmann test for fun
# ggalg = GradientGrassmann(method=OptimKit.ConjugateGradient, maxiter = 100, tol=1e-8, verbosity=3) # can only do this after tagging v0.13



##############################################
# diagonal case

D = 2
V = Vect[A4Object](D0 => D, D1 => D);
inf_init = InfiniteMPS([P], [V]);

ψ, envs = find_groundstate(inf_init, H, VUMPS(verbosity=3, tol=1e-10, maxiter=500));

# other module categories

k = 6 # 4,5,6 gives lapackexception(22), 8,9,10,11 take forever with poorly converged environments, 2,12 has poorly converged envs with imags, 1,3 has imag comps: only 1 and 2 give nonzero
V = Vect[A4Object](A4Object(k,2,i) => 2 for i in 1:MultiTensorKit._numlabels(A4Object, k, 2))
inf_init = InfiniteMPS([P], [V]);
# expectation_value(inf_init, H)

ψ, envs = find_groundstate(inf_init, H, VUMPS(verbosity=3, tol=1e-10, maxiter=10));
expectation_value(ψ, H, envs)

# unitarity test
i = 1
objects = A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i,i]);
count = 0
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
        if !isapprox(F' * F, one(F); atol=1e-9)
            @show a,b,c,d,es,fs F
            count += 1
        end
    end
end
count

# i=1 one thing not unitary
a,b,c,d,es,fs = (A4Object(1, 1, 2), A4Object(1, 1, 2), A4Object(1, 1, 2), A4Object(1, 1, 2), A4Object[A4Object(1, 1, 4), A4Object(1, 1, 1), A4Object(1, 1, 3), A4Object(1, 1, 2)], A4Object[A4Object(1, 1, 4), A4Object(1, 1, 1), A4Object(1, 1, 3), A4Object(1, 1, 2)])

Fblocks = Vector{Any}()
for e in es
    for f in fs
        Fs = Fsymbol(a, b, c, d, e, f)
        @show a,b,c,d,e,f Nsymbol(a,b,e), Nsymbol(e,c,d), Nsymbol(b,c,f), Nsymbol(a,f,d) Fs
        push!(Fblocks,
            reshape(Fs,
                    (size(Fs, 1) * size(Fs, 2),
                    size(Fs, 3) * size(Fs, 4))))
    end
end
Fblocks
Fblocks[end]
F = hvcat(length(fs), Fblocks...)
F * F'

block22 = Fsymbol(a,a,a,a,a,a)
Fblock22 = hvcat(4, block22...)

transpose.(Fblocks)
Ftr = hvcat(length(fs), transpose.(Fblocks)...)

# checking multiplicity
function obs(i::Int)
    return A4Object.(i, i, MultiTensorKit._get_dual_cache(I)[2][i,i])
end

for i in 1:12
    !any(Nsymbol(a,b,c) > 1 for a in obs(i), b in obs(i), c in obs(i))  && @show i
end