# Extending to multifusion category theory

This section will explain how to go from a fusion category to a multifusion category, as well as why one would want to consider the latter. Multifusion categories naturally embed the structure of **bimodule categories**. To explain this, we must start by explaining module categories over fusion categories, following this up with (invertible) bimodule categories, and finishing off with the multifusion structure.

## Module categories
We will use the notation in [Lootens et al.](@cite Lootens_2023) for fusion categories and module categories over these. Starting from a fusion category $\mathcal{D}$ with simple objects $\alpha, \beta, ... \in \mathcal{I}_\mathcal{D}$, we call its associator $F^{\alpha \beta \gamma}: \alpha \otimes (\beta \otimes \gamma) \xrightarrow{\sim} (\alpha \otimes \beta) \otimes \gamma$ the **monoidal associator**. An F-move is now graphically portrayed as

![Fmove_D]()

We can consider the right module category $\mathcal{M}$ over $\mathcal{D}$, which is a category (not necessarily fusion!) with a right action $\triangleleft: \mathcal{M} \times \mathcal{D} \rightarrow \mathcal{M}$ 