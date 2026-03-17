# massive comparison of many multilevel GPU-algorithms including:
* for plates + curved shells
* goal is thickness-ind and h-independent performance
* implement all solvers first in python, then on GPU.
    * can do less work if identify which solvers are thick-ind in python first
    * prefer additive schwarz smooters if using multigrid

* GMG
* AMG: SA-AMG, CF-AMG, RN-AMG
* Domain decomp: FETI, FETI-DP, BDDC (and multilevel variants)
* Multilevel Additive Schwarz (MAS)