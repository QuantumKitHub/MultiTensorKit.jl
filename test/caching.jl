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