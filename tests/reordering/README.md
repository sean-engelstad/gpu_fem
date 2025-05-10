# Reordering of Linear Systems

There are three types of reorderings we consider in this work:
- Reverse Cuthill McKee (RCM): reduces bandwidth of the matrix
- qordering: randomly reorders sections of matrix (pruning size) after RCM to reduce chain lengths
    for more stable ILU preconditioners and GMRES convergence
- AMD: don't really know what this one does, but TACS used it

The general purpose of reorderings is the following (see literature/reordering for more details):
- reduce the bandwidth and number of nonzeroes in the matrix after ILU(k) or full LU fillin
    - results in faster LU and ILU(k) solves
    - less matrix storage required in sparse linear systems
- for qordering, randomness reduces chain lengths which:
    - decreases cascading error in ILU(k) preconditioners or full LU
    - results in more numerically stable preconditioned GMRES and LU solves

## Documentation of Permutations

The permutation maps are the following, it may seem kind of reversed, but this is
common in other codes too. As long as you are consistent it's fine.
- perm: new entry => old entry
- iperm: old entry => new entry

The permutation matrices P and Q are rotation matrices with:
P*Q = Q*P = I and P^-1 = P^T = Q and vice versa
- P equiv to perm : and P(ej) = ei where j is new entry, i is old entry
- Q equiv to iperm: and Q(ei) = ej where i is old entry, j is new entry

Consider the unreordered linear system
   $$ K*u = f $$
Now we pre-multiply by Q or the iperm map to take old rows to new rows
   $$ Q*K*u = Q*f $$
Then we reorder the columns of K and u by multiplying the
identity with $I = Q^T * Q $:
   $$ (Q*K*Q^T) * Q*u = Q*f $$
We now define the reordered linear system with:
- $ K' = QKQ^T $
- $ u' = Q * u$
- $ f' = Q * f$
The reordered linear system is then:
   $$ K' * u' = f' $$
Where the original unreordered solution is obtained by $P$ the perm map:
- $ u = P * u' $