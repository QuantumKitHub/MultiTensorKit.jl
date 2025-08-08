using TensorKitSectors
using MultiTensorKit
using Revise

testobj = A4Object(1, 1, 1) # fusion cat object
unit = one(testobj)
collect(testobj ⊗ unit)
@assert unit == leftone(testobj) == rightone(testobj)

testobj2 = A4Object(2, 2, 1)
unit2 = one(testobj2)
collect(testobj2 ⊗ unit2)
@assert unit2 == leftone(testobj2) == rightone(testobj2)

testmodobj = A4Object(1, 2, 1)
one(testmodobj)
leftone(testmodobj)
rightone(testmodobj)

Fsymbol(testobj, testobj, A4Object(1, 1, 3), testobj, A4Object(1, 1, 3), A4Object(1, 1, 4))

# using Artifacts
# using DelimitedFiles

# artifact_path = joinpath(artifact"fusiondata", "MultiTensorKit.jl-data-v0.1.1")
# filename = joinpath(artifact_path, "A4", "Fsymbol_4.txt")
# txt_string = read(filename, String)
# F_arraypart = copy(readdlm(IOBuffer(txt_string)));

# F_arraypart = MultiTensorKit.convert_Fs(F_arraypart)
# i,j,k,l = (4,12,12,2) # 5,2,8,10
# a,b,c,d,e,f = (1, 11, 3, 1, 1, 3) #(2,1,1,2,2,3)
# testF = F_arraypart[(i,j,k,l)][(a,b,c,d,e,f)]
# a_ob, b_ob, c_ob, d_ob, e_ob, f_ob = A4Object.(((i, j, a), (j, k, b),
#                                                 (k, l, c), (i, l, d),
#                                                 (i, k, e), (j, l, f)))
# result = Array{ComplexF64,4}(undef,
#                             (Nsymbol(a_ob, b_ob, e_ob),
#                             Nsymbol(e_ob, c_ob, d_ob),
#                             Nsymbol(b_ob, c_ob, f_ob),
#                             Nsymbol(a_ob, f_ob, d_ob)))

# map!(result, reshape(testF, size(result))) do pair
#     return pair[2]
# end

N = MultiTensorKit._get_Ncache(A4Object);

duals = MultiTensorKit._get_dual_cache(A4Object)[2]
# checking duals is correct
for i in 1:12, j in 1:12
    for (index, a) in enumerate(duals[i, j])
        aob = A4Object(i, j, index)
        bob = A4Object(j, i, a)
        leftone(aob) ∈ aob ⊗ bob && rightone(aob) ∈ bob ⊗ aob || @show i, j, aob, bob
    end
end

A = MultiTensorKit._get_Fcache(A4Object)
a, b, c, d, e, f = A4Object(1, 1, 3), A4Object(1, 1, 2), A4Object(1, 1, 2),
                   A4Object(1, 1, 2), A4Object(1, 1, 2), A4Object(1, 1, 2)
a, b, c, d, e, f = A4Object(1, 1, 1), A4Object(1, 1, 2), A4Object(1, 1, 2),
                   A4Object(1, 1, 1), A4Object(1, 1, 2), A4Object(1, 1, 4)
coldict = A[a.i, a.j, b.j, c.j]
bla = get(coldict, (a.label, b.label, c.label, d.label, e.label, f.label)) do
    return coldict[(a.label, b.label, c.label, d.label, e.label, f.label)]
end

###### TensorKit stuff ######
using MultiTensorKit
using TensorKit
using Test

# 𝒞 x 𝒞 example

obj = A4Object(2, 2, 1)
obj2 = A4Object(2, 2, 2)
sp = Vect[A4Object](obj => 1, obj2 => 1)
A = TensorMap(ones, ComplexF64, sp ⊗ sp ← sp ⊗ sp)
transpose(A, (2, 4), (1, 3))

blocksectors(sp ⊗ sp)
@plansor fullcont[] := A[a b; a b] # problem here is that fusiontrees for all 12 units are given

# 𝒞 x ℳ example
obj = A4Object(1, 1, 1)
obj2 = A4Object(1, 2, 1)

sp = Vect[A4Object](obj => 1)
sp2 = Vect[A4Object](obj2 => 1)
@test_throws ArgumentError("invalid fusion channel") TensorMap(rand, ComplexF64,
                                                               sp ⊗ sp2 ← sp)
homspace = sp ⊗ sp2 ← sp2
A = TensorMap(ones, ComplexF64, homspace)
fusiontrees(A)
permute(space(A), ((1,), (3, 2)))
transpose(A, (1, 2), (3,)) == A
transpose(A, (3, 1), (2,))

Aop = TensorMap(ones, ComplexF64, conj(sp2) ⊗ sp ← conj(sp2))
transpose(Aop, (1, 2), (3,)) == Aop
transpose(Aop, (1,), (3, 2))

@plansor Acont[a] := A[a b; b] # should not have data bc sp isn't the unit 

spfix = Vect[A4Object](one(obj) => 1)
Afix = TensorMap(ones, ComplexF64, spfix ⊗ sp2 ← sp2)
@plansor Acontfix[a] := Afix[a b; b] # should have a fusion tree

blocksectors(sp ⊗ sp2)
A = TensorMap(ones, ComplexF64, sp ⊗ sp2 ← sp ⊗ sp2)
@plansor fullcont[] := A[a b; a b] # same 12 fusiontrees problem

# completely off-diagonal example

obj = A4Object(5, 4, 1)
obj2 = A4Object(4, 5, 1)
sp = Vect[A4Object](obj => 1)
sp2 = Vect[A4Object](obj2 => 1)
conj(sp) == sp2

A = TensorMap(ones, ComplexF64, sp ⊗ sp2 ← sp ⊗ sp2)
Aop = TensorMap(ones, ComplexF64, sp2 ⊗ sp ← sp2 ⊗ sp)

At = transpose(A, (2, 4), (1, 3))
Aopt = transpose(Aop, (2, 4), (1, 3))

blocksectors(At) == blocksectors(Aop)
blocksectors(Aopt) == blocksectors(A)

@plansor Acont[] := A[a b; a b] # ignore this error for now
@plansor Acont2[] := A[b a; b a]

testsp = SU2Space(0 => 1, 1 => 1)
Atest = TensorMap(ones, ComplexF64, testsp ⊗ testsp ← testsp ⊗ testsp)
@plansor Aconttest[] := Atest[a b; a b]

# 𝒞 x ℳ ← ℳ x 𝒟
c = A4Object(1, 1, 1)
m = A4Object(1, 2, 1)
d = A4Object(2, 2, 1)
W = Vect[A4Object](c => 1) ⊗ Vect[A4Object](m => 1) ←
    Vect[A4Object](m => 1) ⊗ Vect[A4Object](d => 1)

# bram stuff
using TensorKitSectors
for i in 1:12, j in 1:12
    for a in A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i, j])
        F = Fsymbol(a, dual(a), a, a, leftone(a), rightone(a))[1, 1, 1, 1]
        isapprox(F, frobeniusschur(a) / dim(a); atol=1e-15) ||
            @show a, F, frobeniusschur(a) / dim(a) # check real
        isreal(frobeniusschur(a)) || isapprox(abs(frobeniusschur(a)), 1.0; atol=1e-15) ||
            @show a, frobeniusschur(a), abs(frobeniusschur(a))
    end
end

for i in 1:12, j in 1:12 # 18a
    i != j || continue
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i, j])
    @assert all(dim(m) > 0 for m in objsij)
end

for i in 1:12, j in 1:12 # 18b
    objsii = A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i, i])
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i, j])

    Ndict = Dict{Tuple{A4Object,A4Object,A4Object},Int}()
    for a in objsii, m in objsij
        for n in a ⊗ m
            Ndict[(a, m, n)] = Nsymbol(a, m, n)
        end
    end

    for a in objsii, m in objsij
        isapprox(dim(a) * dim(m), sum(Ndict[(a, m, n)] * dim(n) for n in a ⊗ m);
                 atol=2e-9) || @show a, m
    end
end

for i in 1:12, j in 1:12 # 18c
    objsii = A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i, i])
    objsij = A4Object.(i, j, MultiTensorKit._get_dual_cache(A4Object)[2][i, j])
    m_dimsum = sum(dim(m)^2 for m in objsij)
    c_dimsum = sum(dim(c)^2 for c in objsii)
    isapprox(m_dimsum, c_dimsum; atol=1e-8) || @show i, j, c_dimsum, m_dimsum
end

(a, b, c, d, e, f) = (A4Object(2, 1, 1), A4Object(1, 2, 1), A4Object(2, 2, 11),
                      A4Object(2, 2, 11), A4Object(2, 2, 9), A4Object(1, 2, 1))
Fsymbol(a, b, c, d, e, f)
zeros(ComplexF64, Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d)) ==
Fsymbol(a, b, c, d, e, f)

# testing blocksectors 
using BlockTensorKit
W = Vect[A4Object](A4Object(2, 2, 12) => 1) ←
    ProductSpace{GradedSpace{A4Object,NTuple{486,Int64}},0}()
W isa TensorMapSpace{GradedSpace{A4Object,NTuple{486,Int64}}}
W isa HomSpace
W isa HomSpace{S} where {S<:SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}}}
W isa TensorMapSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}}}

# this appears as well (N1=N2=0)
W = ProductSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}},0}() ←
    ProductSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}},0}()
typeof(W)
W isa TensorMapSpace{S} where {S<:GradedSpace{A4Object,NTuple{486,Int64}}}
W isa HomSpace{S} where {S<:GradedSpace{A4Object,NTuple{486,Int64}}}
W isa HomSpace
W isa TensorMapSpace

W isa TensorMapSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}},0,0}
W isa TensorMapSpace{GradedSpace{A4Object,NTuple{486,Int64}},0,0}
TensorMapSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}},0,0} <:
TensorMapSpace{GradedSpace{A4Object,NTuple{486,Int64}},N₁,N₂} where {N₁,N₂}
W isa TensorSpace{SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}}}

GradedSpace{A4Object,NTuple{486,Int64}} <:
BlockTensorKit.SumSpace{GradedSpace{A4Object,NTuple{486,Int64}}}

############ MPSKit wow ############
using MultiTensorKit
using TensorKit
using MPSKit, MPSKitModels
C1 = A4Object(1, 1, 1)
C0 = A4Object(1, 1, 4) # unit
M = A4Object(1, 2, 1)
D0 = A4Object(2, 2, 12) # unit
D1 = A4Object(2, 2, 2) # self-dual object
collect(D0 ⊗ D1)
collect(D1 ⊗ D1)

P = Vect[A4Object](D0 => 1, D1 => 1)
h = TensorMap(ones, ComplexF64, P ⊗ P ← P ⊗ P)

using Profile
Profile.init(; delay=0.1)
lattice = InfiniteChain(1)
t = time()
H = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(lattice)); # 15min, 10.5min
dt = time() - t
println("Time to create Hamiltonian: ", dt, " seconds")

# testing insertleft/rightunit
sp = SU2Space(0 => 1, 1 => 1)
ht = TensorMap(ones, ComplexF64, P ← P)
htl = TensorMap(ones, ComplexF64, P ← one(P))
htr = TensorMap(ones, ComplexF64, one(P) ← P)
htnone = TensorMap(ones, ComplexF64, one(P) ← one(P))
insertrightunit(htr) # adding to empty space
insertleftunit(htr)
insertrightunit(htl)
insertleftunit(htl) # adding to empty space
insertrightunit(htnone)
insertleftunit(htnone)

D = 4
V = Vect[A4Object](M => D);
t = time()
inf_init = InfiniteMPS([P], [V]); # 8min, 6.5min
dt = time() - t
println("Time to create InfiniteMPS: ", dt, " seconds")

# VUMPS
t = time()
ψ, envs = find_groundstate(inf_init, H, VUMPS(; verbosity=3, tol=1e-12, maxiter=20)); # 40 min, 30min
dt = time() - t
println("Time to find groundstate: ", dt, " seconds")
expectation_value(ψ, H, envs)

using JLD2
# jldsave("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testH.jld2"; H)
# jldsave("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testinf_init.jld2"; inf_init)
# jldsave("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testgs.jld2"; ψ)
# jldsave("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testenvs.jld2"; envs)

ψ = jldopen("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testgs.jld2", "r") do file
    return read(file, "ψ")
end
H = jldopen("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testH.jld2", "r") do file
    return read(file, "H")
end
inf_init = jldopen("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testinf_init.jld2",
                   "r") do file
    return read(file, "inf_init")
end
envs = jldopen("C:/Boris/Unief/PhD/Code/MultiTensorKit/Saves/testenvs.jld2", "r") do file
    return read(file, "envs")
end

entropy(ψ)
entanglement_spectrum(ψ)
transfer_spectrum(ψ; sector=C0)
correlation_length(ψ; sector=C0)
norm(ψ)

# test correlator

#IDMRG

ψ, envs = find_groundstate(inf_init, H, IDMRG(; verbosity=3, tol=1e-8, maxiter=15));
expectation_value(ψ, H, envs)

#IDMRG2
inf_init2 = InfiniteMPS([P, P], [V, V])
H2 = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(InfiniteChain(2)));
idmrg2alg = IDMRG2(; verbosity=3, tol=1e-8, maxiter=15, trscheme=truncdim(10))
ψ2, envs2 = find_groundstate(inf_init2, H2, idmrg2alg);
expectation_value(ψ2, H2, envs2)

#QuasiParticleAnsatz

momenta = range(0, 2π, 3)
t = time()
excE, excqp = excitations(H, QuasiparticleAnsatz(; ishermitian=true), momenta, ψ, envs;
                          sector=C0, num=1, parallel=false); # not working for some reason
dt = time() - t
# 15min, 100min
println("Time to create excitations: ", dt, " seconds")

#######################################################
# debugging QPA for infinite systems
momentum = momenta[begin]
ϕ₀ = LeftGaugedQP(rand, ψ, ψ; sector=C0, momentum=momenta[begin]);
E = MPSKit.effective_excitation_renormalization_energy(H, ϕ₀, envs, envs)
H_eff = MPSKit.EffectiveExcitationHamiltonian(H, envs, envs, E); # function that acts on QP
alg = QuasiparticleAnsatz(; ishermitian=true)

# do block with eigsolve effectively doing this
my_operator(ϕ) = H_eff(ϕ; alg.alg_environments...)
Es, ϕs, convhist = eigsolve(my_operator, ϕ₀, 1, :SR, alg.alg)

# goes back in the eigsolve during KrylovKit.initialise 
iter = LanczosIterator(my_operator, ϕ₀, alg.alg.orth);
fact = initialize(iter; verbosity=alg.alg.verbosity)

# goes wrong during apply of KrylovKit.initialise
Ax₀ = KrylovKit.apply(iter.operator, iter.x₀)

# apply(f,x) = f(x) calculates EffectiveExcitationHamiltonian(H, envs, envs, E) of QP
iter.operator(iter.x₀)
MPSKit.EffectiveExcitationHamiltonian(H, envs, envs, E)(ϕ₀; alg.alg_environments...)
H_eff(ϕ₀; alg.alg_environments...)

# error occurs when calculating the environment 
qp_envs = environments(ϕ₀, H, envs, envs; alg.alg_environments...)

# goes wrong when calculating environments of QP
lBs = PeriodicVector([MPSKit.allocate_GBL(ϕ₀, H, ϕ₀, i) for i in 1:length(ϕ₀)]);
MPSKit.left_excitation_transfer_system(lBs[1], H, ϕ₀; solver=MPSKit.Defaults.linearsolver)

# problem occurs at the linsolve which calls GMRES
found = zerovector(lBs[1])
H_partial = map(h -> getindex(h, 1:1, 1, 1, 1:1), parent(H))
T = TransferMatrix(exci.right_gs.AR, H_partial, exci.left_gs.AL)
start = scale!(last(found[1:1] * T), cis(-momenta[begin] * 1))
if exci.trivial && isidentitylevel(H, i)
    # not using braiding tensors here, leads to extra leg
    util = similar(exci.left_gs.AL[1], space(parent(H)[1], 1)[1])
    fill_data!(util, one)
    @plansor start[-1 -2; -3 -4] -= start[2 1; -3 3] *
                                    util[1] *
                                    r_RL(exci.right_gs)[3; 2] *
                                    l_RL(exci.right_gs)[-1; -4] *
                                    conj(util[-2])
end
found[1] = add!(start, lBs[1])

T = TransferMatrix(exci.right_gs.AR, exci.left_gs.AL)
if exci.trivial
    # deal with extra leg
    @plansor lRL_util[-1 -2; -3] := l_RL(exci.right_gs)[-1; -3] * conj(util[-2])
    @plansor rRL_util[-1 -2; -3] := r_RL(exci.right_gs)[-1; -3] * util[-2]
    T = regularize(T, lRL_util, rRL_util)
end

found[1], convhist = linsolve(flip(T), found[1], found[1], MPSKit.Defaults.linearsolver, 1,
                              -cis(-momenta[begin] * 1))

##############################################################################
# quick test on complex f symbols and dimensions
testp = Vect[A4Object](one(A4Object(i, i, 1)) => 1 for i in 1:12)
dim(testp)
oneunit(testp)

# finite stuff
L = 10
lattice = FiniteChain(L)
P = Vect[A4Object](D0 => 1, D1 => 1)
D = 4
V = Vect[A4Object](M => D)

dmrgalg = DMRG(; verbosity=3, tol=1e-8, maxiter=100,
               alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false))
fin_init = FiniteMPS(L, P, V; left=V, right=V)
Hfin = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(lattice));
open_boundary_conditions(H, L) == Hfin
ψfin, envsfin = find_groundstate(fin_init, Hfin, dmrgalg);
expectation_value(ψfin, Hfin, envsfin) / (L - 1)

entropy(ψfin, round(Int, L / 2))
entanglement_spectrum(ψfin, round(Int, L / 2))
Es, states, convhist = exact_diagonalization(Hfin; sector=D0);
Es / (L - 1)

dmrg2alg = DMRG2(; verbosity=3, tol=1e-8, maxiter=15,
                 alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false),
                 trscheme=truncdim(10))
ψfin2, envsfin2 = find_groundstate(fin_init, Hfin, dmrg2alg);
0expectation_value(ψfin2, Hfin, envsfin2) / (L - 1)

entropy(ψfin2, round(Int, L / 2))
entanglement_spectrum(ψfin2, round(Int, L / 2))

S = left_virtualspace(Hfin, 1)
oneunit(S)
eltype(S)
oneunit(eltype(S)) # should error

# excitations
qpaalg = QuasiparticleAnsatz(; ishermitian=false, tol=1e-8, maxiter=100)
excEfin, excqpfin = excitations(Hfin, qpaalg, ψfin, envsfin; sector=C0, num=1);
excEfin

excFIN, excqpFIN = excitations(Hfin, FiniteExcited(; gsalg=dmrg2alg), ψfin; num=1);
excFIN

# changebonds test
dim(left_virtualspace(ψ, 1))
ψch, envsch = changebonds(ψ, H, OptimalExpand(; trscheme=truncerr(1e-3)), envs)
dim(left_virtualspace(ψch, 1))

# time evolution

ψt, envst = timestep(ψ, H, 3, 0,
                     TDVP(; integrator=MPSKit.Defaults.alg_expsolve(; ishermitian=true)),
                     envs);
et = expectation_value(ψt, H, envst)
e = expectation_value(ψ, H, envs)
isapprox(et, e; atol=1e-12) # hermitian

tdvpalg = TDVP(; integrator=MPSKit.Defaults.alg_expsolve(; ishermitian=true, krylovdim=10))
ψt, envst = time_evolve(ψ, H, range(0, 1, 10), tdvpalg, envs)

timealg = WII(; tol=1e-8, maxiter=100)
make_time_mpo(H, 0.1, timealg)

# stat-mech stuff
mpo = InfiniteMPO([h])
ψ, envs = leading_boundary(inf_init, mpo,
                           VUMPS(; verbosity=3, tol=1e-8, maxiter=15,
                                 alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;
                                                                           ishermitian=false)));

# addition, substraction, multiplication

# finite 
Hfin * ψfin

# infinite
H * ψ # currently doesn't work
MPSKit.DenseMPO(H) * ψ # does work

# approximate
ψa, _ = approximate(ψ, (mpo, inf_init), IDMRG());

mpo2 = InfiniteMPO([h, h])
inf_init2 = InfiniteMPS([P, P], [V, V])
H2 = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(InfiniteChain(2)));
ψ2, envs2 = find_groundstate(inf_init2, H2, VUMPS(; verbosity=3, tol=1e-10, maxiter=15));
ψa, _ = approximate(ψ2, (mpo2, inf_init2), IDMRG2(; trscheme=truncdim(10)));

# # testing InfiniteMPOHamiltonian and FiniteMPOHamiltonian constructor not relying on MPSKitModels

sp = Vect[FibonacciAnyon](:I => 1, :τ => 1)
t = TensorMap(ones, ComplexF64, sp ← sp)
InfiniteMPOHamiltonian(PeriodicArray([sp]), i => t for i in 1:1)
H = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(InfiniteChain(1)));

##############################################
# diagonal case

D = 4
V = Vect[A4Object](D0 => D, D1 => D);
inf_init = InfiniteMPS([P], [V]);

ψ, envs = find_groundstate(inf_init, H, VUMPS(; verbosity=3, tol=1e-10, maxiter=500));

# other module categories
# 4,5,7,8,9,10,11,12 gives poorly converged envs,
k = 6 # 6 gives lapackexception(22)
V = Vect[A4Object](A4Object(k, 2, i) => 2
                   for i in 1:MultiTensorKit._numlabels(A4Object, k, 2))
inf_init = InfiniteMPS([P], [V]);
# expectation_value(inf_init, H)

ψ, envs = find_groundstate(inf_init, H, VUMPS(; verbosity=3, tol=1e-10, maxiter=10));
expectation_value(ψ, H, envs)

# checking multiplicity
function obs(i::Int)
    return A4Object.(i, i, MultiTensorKit._get_dual_cache(A4Object)[2][i, i])
end

for i in 1:12
    !any(Nsymbol(a, b, c) > 1 for a in obs(i), b in obs(i), c in obs(i)) && @show i
end

# trying to make heisenberg
using MultiTensorKit, TensorKit
using MPSKit, MPSKitModels

D1 = A4Object(1, 1, 2) # 3-dimensional irrep of A4
M = A4Object(2, 1, 1) # Vec

P = Vect[A4Object](D1 => 1)
h_aux1 = TensorMap(ones, ComplexF64, P ← P ⊗ P)
h_aux2 = TensorMap(ones, ComplexF64, P ⊗ P ← P)

@plansor h[-1 -2; -3 -4] := h_aux2[-1 1; -3] * h_aux1[-2; 1 -4] # different basis
lattice = InfiniteChain(1)
t = time()
H = @mpoham -sum(h{i,j} for (i, j) in nearest_neighbours(lattice));
dt = time() - t
println("Time to create Hamiltonian: ", dt, " seconds")

D = 4
V = Vect[A4Object](M => D)
t = time()
inf_init = InfiniteMPS([P], [V]);
dt = time() - t
println("Time to create InfiniteMPS: ", dt, " seconds")

# VUMPS
t = time()
ψ, envs = find_groundstate(inf_init, H, VUMPS(; verbosity=3, tol=1e-12, maxiter=200));
dt = time() - t
println("Time to find groundstate: ", dt, " seconds")
expectation_value(ψ, H, envs) # this gives 0 oopsie

### caching checks
length(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE)

length(TensorKit.treepermutercache) # tensor stuff
length(TensorKit.treetransposercache)
length(TensorKit.treebraidercache)

length(TensorKit.transposecache) # fusion tree stuff
length(TensorKit.braidcache)
