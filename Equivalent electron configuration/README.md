Determines all possible atomic configurations for $`n`$ given equivalent electrons, i.e. electrons having the same orbital angular momentum $`l`$.

Produces a Slater's table which might look as such (for two $`l=p`$ electrons):

| $`{M_l}/{M_s}`$ |  $`-1`$  |   $`0`$  |   $`1`$  |
|------|------|------|------|
|  $`2`$   |   $`0`$  |   $`1`$  |  $`0`$   |
|  $`1`$   |   $`1`$  |   $`2`$  |  $`1`$   |
|  $`0`$   |   $`1`$  |   $`3`$  |  $`1`$   |

 Here the top row, $`m_s`$ are the allowed net spin of the atom, first column, $`m_l`$ are the allowed net orbital angular momentum values. The table values are the number of ways to arrange the electrons for that net $`m_l,\ m_s`$ configuration. This also takes into account Pauli's exclusion principle.

 Also outputs the term symbols, $`{}^{2S+1}L_{J}`$ of the allowed final states in increasing order of energy as per Hund's rules.

 Outputs of the form $`3P ( 0. 1. 2. )`$ refers to $`3`$ term symbols in increasing order of energy, i.e. $`E({}^3P_0)<E({}^3P_1)<E({}^3P_2)`$
