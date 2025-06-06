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

## Multifusion categories
A fusion category has the important condition that $\mathsf{End}_\mathcal{C}(1_\mathcal{C}) \cong \mathbb{C}$, i.e., the unit of the fusion category is simple. If we drop this condition, then we consider a **multifusion category**. We assume the multifusion category itself to be indecomposable, meaning it is not the direct sum of two non-trivial multifusion categories. Let us call this multifusion category $\mathcal{C}$. It will be clear in a moment that this will not be ambiguous. Its unit can then be decomposed as
$$ 1_\mathcal{C} = \bigoplus_{i=1}^r 1_r$$,
i.e., it is decomposed into simple objects of the 
### Why opposite module categories end up being necessary in MultiTensorKit
something something B-move of module leg