# Extending to multifusion category theory

This section will explain how to go from a fusion category to a multifusion category, as well as why one would want to consider the latter. Multifusion categories naturally embed the structure of **bimodule categories**. To explain this, we must start by explaining module categories over fusion categories, following this up with (invertible) bimodule categories, and finishing off with the multifusion structure.

## Module categories
We will mostly use the notation in [Lootens et al.](@cite Lootens_2023) for fusion categories and module categories over these. We start from a fusion category $\mathcal{D}$ defined by the triple $(\otimes_\mathcal{D}, \mathbb{1}_\mathcal{D}, {}^\mathcal{D}\!F)$ with simple objects $\alpha, \beta, ... \in \mathcal{I}_\mathcal{D}$. We drop the $\mathcal{D}$ subscript when there is no ambiguity concerning the fusion category. We call its associator 
$${}^\mathcal{D}\!F^{\alpha \beta \gamma}: \alpha \otimes (\beta \otimes \gamma) \rightarrow (\alpha \otimes \beta) \otimes \gamma$$
 the **monoidal associator**.  An F-move is now graphically portrayed as:

![Fmove_D]()

We can consider the **right module category** $\mathcal{M}$ over $\mathcal{D}$, which is a category (not necessarily fusion!) with (isomorphism classes of) simple objects $\mathcal{I}_\mathcal{M} = \{A,B,...\}$, a right action 
$$\triangleleft: \mathcal{M} \times \mathcal{D} \rightarrow \mathcal{M}$$
and the **right module associator** 
$${}^\triangleleft\!F^{A\alpha\beta}: A \triangleleft (\alpha \otimes \beta) \rightarrow (A \triangleleft \alpha) \triangleleft \beta.$$
An F-move with this module associator can be expressed as:

![Fmove_MD]()

The module structure of $\mathcal{M}$ is now defined as the triple $(\mathcal{M}, \triangleleft, {}^\triangleleft\!F)$. The right module associator now satisfies a mixed pentagon equation with ${}^\mathcal{D}\!F$. 

Similarly, we can define a **left module category** $(\mathcal{M}, \triangleright, {}^\triangleright\!F)$ over a fusion category $(\otimes_\mathcal{C}, 1_\mathcal{C}, {}^\mathcal{C}\!F)$. The functor is now a left action of $\mathcal{C}$ on $\mathcal{M}$ 
$$\triangleright: \mathcal{C} \times \mathcal{M} \rightarrow \mathcal{M},$$
while the **left module associator** is a natural isomorphism that acts as
$${}^\triangleright\!F^{abA}: (a \otimes b) \triangleright A \rightarrow a \triangleright (b \triangleright A)$$
for $\mathcal{I}_\mathcal{C} = \{a,b,...\}$. The left module associator also fulfills a mixed pentagon equation with ${}^\mathcal{C}\!F$. An F-move with ${}^\mathcal{C}\!F$ takes on the form:

![Fmove_CM]()

We can combine the concepts of left and right module categories as follows. Say there are two fusion categories $\mathcal{C}$ and $\mathcal{D}$. A $(\mathcal{C}, \mathcal{D})$-**bimodule category** is a module category, now defined through a sextuple $(\mathcal{M}, \triangleright, \triangleleft, {}^\triangleright\!F, {}^\triangleleft\!F, {}^{{\triangleright \hspace{-1.2mu}\triangleleft}}\!F)$ such that $(\mathcal{M}, \triangleright, {}^\triangleright\!F)$ is a left $\mathcal{C}$-module category, and $(\mathcal{M}, \triangleleft, {}^\triangleleft\!F)$ is a right $\mathcal{D}$-module category, and with additional structure such that the **bimodule associator** acts as 
$${}^{{\triangleright \hspace{-1.2mu}\triangleleft}}\!F^{aA\alpha}: (a \triangleright A) \triangleleft \alpha \rightarrow a \triangleright (A \triangleleft \alpha)$$
for $a \in \mathcal{I}_\mathcal{C}, \alpha \in \mathcal{I}_\mathcal{D}, A \in \mathcal{I}_\mathcal{M}$. The bimodule associator fulfills a mixed pentagon equation with the module associators. An F-move with ${}^{{\triangleright \hspace{-1.2mu}\triangleleft}}\!F$ is given by:

![Fmove_CMD]()
## Opposite module categories
Consider a fusion category $\mathcal{D}$ and a *right* module category $\mathcal{M}$ over $\mathcal{D}$. We can define $\mathcal{M}^{\text{op}}$ to be the opposite category of $\mathcal{M}$ [etingof2009](@cite). Then $\mathcal{M}^{\text{op}}$ is a *left* module category over $\mathcal{D}$. A similar statement can be made starting from a left module category and getting an opposite right module category. In particular, given a $(\mathcal{C}, \mathcal{D})$-bimodule category $\mathcal{M}$ over the fusion categories $\mathcal{C}, \mathcal{D}$, we see immediately that $\mathcal{M}^{\text{op}}$ is a $(\mathcal{D}, \mathcal{C})$-bimodule category.

Interestingly, due to the opposite actions
$$\triangleleft^{\text{op}}: \mathcal{D} \times \mathcal{M}^{\text{op}} \rightarrow \mathcal{M}^{\text{op}}$$ 
and 
$$\triangleright^{\text{op}}: \mathcal{M}^{\text{op}} \times \mathcal{C} \rightarrow \mathcal{M}^{\text{op}},$$
there is a valid notion of multiplying a module category with its opposite:
$$
\mathcal{M} \times \mathcal{M}^\text{op} \rightarrow \mathcal{C}, \quad \mathcal{M}^\text{op} \times \mathcal{M} \rightarrow \mathcal{D}.
$$

something something morita equivalence and invertible bimodule, maybe drinfeld center, maybe anyons, maybe domain walls

## Multifusion categories
A fusion category has the important condition that $\mathsf{End}_\mathcal{C}(1_\mathcal{C}) \cong \mathbb{C}$, i.e., the unit of the fusion category is simple. If we drop this condition, then we consider a **multifusion category**. We assume the multifusion category itself to be indecomposable, meaning it is not the direct sum of two non-trivial multifusion categories. Let us call this multifusion category $\mathcal{C}$. It will be clear in a moment that this will not be ambiguous. Its unit can then be decomposed as
$$ 1_\mathcal{C} = \bigoplus_{i=1}^r 1_r,$$
i.e., it is decomposed into simple unit objects of the subcategories $\mathcal{C}_{ij} \coloneqq 1_i \otimes \mathcal{C} \otimes 1_j$. With this, we see that the multifusion category itself can be decomposed into its subcategories
$$\mathcal{C} = \bigoplus_{i,j=1}^r \mathcal{C}_{ij}.$$
We call this an $r \times r$ multifusion category. 

We want to consider multifusion categories because **their structure encapsulates that of (bi-)module categories**. Every diagonal category $\mathcal{C}_{ii} \coloneqq \mathcal{C}_i$ is a fusion category, and every off-diagonal category $\mathcal{C}_{ij}$ is an invertible $(\mathcal{C}_{i}, \mathcal{C}_{j})$-bimodule category. That way, as long as we know how the simple objects of the fusion and module categories fuse with one another, and we can determine all the monoidal and module associators, we can treat the multifusion category as one large fusion category with limited fusion rules. In particular, the tensor product 
$$\otimes_\mathcal{C}: \mathcal{C}_{ij} \times \mathcal{C}_{kl} \rightarrow \delta_{jk}\mathcal{C}_{il}$$
takes on the same structure as the product of two matrix units. This is not a coincidence; there is a deep relation between multifusion categories and matrix algebras [etingof2016tensor; Section 4.3](@cite). 

Given a subcategory $\mathcal{C}_{ij}$, we can define the **left** unit as the unit object of the fusion category $\mathcal{C}_i$, while the **right** unit is the unit object of the fusion category $\mathcal{C}_j$. In other words, the left unit of $\mathcal{C}_{ij}$ is the unique object of the multifusion category $\mathcal{C}$ for which
$$1_i \otimes_\mathcal{C} M_{ij} = M_{ij} \quad \forall M_{ij} \in \mathcal{C}_{ij},$$
and similarly for the right unit of $\mathcal{C}_{ij}$,
$$M_{ij} \otimes_\mathcal{C} 1_j = M_{ij} \quad \forall M_{ij} \in \mathcal{C}_{ij}.$$

We can also immediately see that for a (bi)module subcategory $\mathcal{C}_{ij}$, the opposite (bi)module subcategory $\mathcal{C}_{ij}^{\text{op}} \equiv \mathcal{C}_{ji}$, and as expected,
$$\mathcal{C}_{ij} \times \mathcal{C}_{ji} \rightarrow \mathcal{C}_i,$$
just like what we concluded when considering opposite module categories outside of the multifusion structure.
### 2-category and coloring
diagrammatic calculus with coloring and such and so

Multifusion categories can also be interpreted as 2-categories. We still interpret the objects of this 2-category the same way. The 1-morphisms are the subcategories themselves, and the 2-morphisms the morphisms of the multifusion category. The graphical calculus of monoidal 1-categories can be extended to 2-categories by use of *colorings*. We have previously differed between module strands and fusion strands by the color of the strand itself. However, in 2-categories the strands (1-morphisms) separate regions which are colored based on the objects they are representing. Since we draw the strands vertically, a single strand results in a left and right region, and the colorings will determine the fusion category which fuses from the left or right with that single strand. In particular, fusion strands necessarily have the same coloring on the left and right, while module strands have a mismatching coloring. 

The simplest non-trivial fusion diagram is a trivalent junction:

![Nsymbol_coloring]()

The most general case is the top left figure, where all three regions have a different coloring. The top middle region having the same coloring from the top left and top right strands follow from the delta function in the tensor product definition. However, as will be explained more in detail later, this most general trivalent junction with three colorings will never be needed. In short, we will always be considering a single bimodule category $\mathcal{C}_{ij}$ at a time, and the only other non-diagonal subcategory which fuses with this is its opposite $\mathcal{C}_{ji}$. This is displayed in the top middle and right. Similarly, two colorings are required when considering the fusion between a fusion and module strand, shown in the bottom left and middle figure. The simplest trivalent junctions boil down to fusions within fusion categories, which is obviously drawn with just one color. This is shown in the bottom right.

### Why opposite module categories end up being necessary in MultiTensorKit
something something B-move of module leg

One of the common manipulations which can act on a tensor map is the transposition of vector spaces. We will refer to this as the bending of legs. One of the elementary bends is the **right bend**, where one of the tensor legs is bent along the right from the codomain to the domain, or vice versa. At the level of the tensor, a covariant index becomes contravariant, or vice versa. Similarly, a **left bend** can also be performed, bending the leg along the left. This guarantees that legs will not cross, preventing braidings which require extra data known as R-symbols. 

Linear algebra tells us that given a (finite-dimensional) vector space $V$ with a basis denoted $\{|e_i\rangle\}$, one can consider the **dual** vector space $V^*$, whose dual basis $\{\langle e_i^*|\}$ satisfies the property $\langle e_j^* | e_i \rangle = \delta_{ij}$. In the diagrammatic calculus, specifying whether a tensor map leg represents a vector space or its dual is done with an arrow. Following the TensorKit convention, legs with arrows pointing downwards are vector spaces, and arrows pointing upwards state that we are considering its dual. In particular, at the level of fusion trees we can also draw arrows on the strands to denote whether we are considering morphisms between objects or dual objects. 

In principle, choosing to bend e.g. codomain legs to the right and domain legs to left is an arbitrary choice, but would require to distinguish between left and right transposes. However, TensorKit.jl is implemented in a way that does not differentiate the two. In particular, we do not have to worry about this when considering categorical symmetries where, in principle, the left and right dual of an object are not equivalent. This is because this left-right symmetry is guaranteed when considering unitary fusion categories, which is what is done in TensorKit and necessarily in MultiTensorKit. 

For this reason, at the level of the fusion trees the topological move that is performed to bend the legs along the right is called a **B-move**. Graphically, one can show that this bend boils down to a particular F-move. The typical equation found in the literature is the following:
![Bmove_lit]()

... The reason to only consider B-moves is rooted in the choice of canonical form of fusion trees within TensorKit, where fusions are iterated over from left to right and splittings from right to left. 

Importantly, we identify the dual vector space labeled by a module category with a vector space labeled by the opposite module category. Consequently,
$$\mathcal{M}^* \simeq \mathcal{M}^\text{op}.$$
In the multifusion setting, this can also be seen graphically. By keeping track of the colorings and the directions of the arrows of the legs, one can see that we need to slightly modify the expression for the B-move to the following:

![Bmove_MF]()

where by $\mathbb{1}_a$ we mean the right unit of $a$ (the left unit we would denote $^{}_a \mathbb{1}$). 

Besides the B-move (and closely related A-move, which we do not illustrate), we can also see how the quantum dimension and Frobenius-Schur indicator expressions get modified. We already know that an F-move of the form $F^{a \bar{a} a}_{a}$ needs to be evaluated for these topological data. Graphically, we find that

![qdim_fs_MF]()

need to show other changed expressions like A-move, dimension, frobenius-schur indicator, what else outside of TensorKitSectors in terms of fusion tree manipulations?

no figures up till now with arrows, will this even be necessary? maybe if we show B-moves with M->Mop

## Examples of multifusion categories
2x2 thing that is isomorphic to ising
generalisation to Tambara-Yamagami

where to say something about rewriting mpskit to be planar such that braidings with module legs are avoided?

Without specifying any of the categories, the simplest non-trivial multifusion category is a $2\times 2$ one, and the categories can be organised in a matrix as
$$\mathcal{C} = \begin{pmatrix} \end{pmatrix}$$