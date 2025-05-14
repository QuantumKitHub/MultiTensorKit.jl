using MultiTensorKit
using TensorKit
using MPSKit, MPSKitModels

C1 = A4Object(1,1,1)
C0 = A4Object(1,1,4) # unit
M = A4Object(1,2,1)
D0 = A4Object(2,2,12) # unit
D1 = A4Object(2,2,2) # self-dual object

P = Vect[A4Object](D0 => 1, D1 => 1)
h = TensorMap(ones, ComplexF64, P ⊗ P ← P ⊗ P)

lattice = InfiniteChain(1)
H = @mpoham -sum(h{i,j} for (i,j) in nearest_neighbours(lattice));

D = 4
V = Vect[A4Object](M => D);
inf_init = InfiniteMPS([P], [V]);

ψ, envs = find_groundstate(inf_init, H, VUMPS(verbosity=3, tol=1e-12, maxiter=20));

# basic TensorKit tests

# diagonal 
obj = A4Object(2,2,1)
obj2 = A4Object(2,2,2)
sp = Vect[A4Object](obj=>1, obj2=>1)
A = TensorMap(ones, ComplexF64, sp ⊗ sp ← sp ⊗ sp)
transpose(A, (2,4,), (1,3,))

blocksectors(sp ⊗ sp)
@plansor fullcont[] := A[a b;a b] # 12 fusiontrees

# 𝒞 x ℳ
obj = A4Object(1,1,1)
obj2 = A4Object(1,2,1)

sp = Vect[A4Object](obj=>1)
sp2 = Vect[A4Object](obj2=>1)
TensorMap(rand, ComplexF64, sp ⊗ sp2 ← sp) # should throw ArgumentError
homspace = sp ⊗ sp2 ← sp2
A = TensorMap(ones, ComplexF64, homspace)
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
@plansor fullcont[] := A[a b;a b] # 12 fusiontrees