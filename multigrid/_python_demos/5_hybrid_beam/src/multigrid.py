

import numpy as np
import scipy as sp

def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, dof_per_node:int=2):
    """
    Perform Block Gauss-Seidel smoothing for 6 DOF per node.
    A: csr_matrix of size (6*nnodes, 6*nnodes)
    b: RHS vector (6*nnodes,)
    x0: initial guess (6*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    ndof = dof_per_node
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x

def block_gauss_seidel_smoother(A, x, defect:np.ndarray, num_iter:int=1, dof_per_node:int=2):
    # do the steps of the smoother
    dx = block_gauss_seidel(A, defect.copy(), x0=np.zeros_like(defect), num_iter=num_iter, dof_per_node=dof_per_node)
    new_x = x + dx
    new_defect = defect - A.dot(dx)
    return new_x, new_defect

def vcycle_solve(grids:list, nvcycles:int=100, pre_smooth:int=1, post_smooth:int=1, debug_print:bool=False):
    # grids are just assembler objects usually (no unified GRID object)
    nlevels = len(grids)
    dof_per_node = grids[0].dof_per_node # get from finest grid assembler

    solns = [np.zeros_like(grids[i].force) for i in range(nlevels)]
    defects = [grids[i].force.copy() for i in range(nlevels)]
    mats = [grids[i].Kmat.copy() for i in range(nlevels)]

    init_defect_norm = np.linalg.norm(defects[0])
    print(f"V-cycle multigrid solve with {nlevels} grids:")
    print(f"\n\t{init_defect_norm=:.2e}\n----------------\n")
    converged = False

    for i_cycle in range(nvcycles):

        # smooth and restrict downwards
        for i in range(0, nlevels - 1):
            # pre-smooth
            if debug_print: print(f"\tpre-smooth grid[{i}]")
            solns[i], defects[i] = block_gauss_seidel_smoother(mats[i], solns[i], defects[i], num_iter=pre_smooth, dof_per_node=dof_per_node)

            # # plot the defect after smoothing
            # grids[i].u = defects[i].copy()
            # grids[i].plot_disp(idof=1)

            # restrict
            if debug_print: print(f"\trestrict grids [{i}]=>[{i+1}]")
            defects[i+1] = grids[i+1].restrict_defect(defects[i])
            solns[i+1] *= 0.0 # resets coarse soln when you restrict defect

            # print(F"{i=} {solns[i].shape=}")

        # coarse grid solve
        if debug_print: print(f"\tcoarse solve on grid[{nlevels-1}]")
        solns[nlevels-1] = sp.sparse.linalg.spsolve(mats[nlevels-1].copy(), defects[nlevels-1])

        # prolong and post-smooth
        for i in range(nlevels-2, -1, -1):

            # proposed prolongate
            if debug_print: print(f"\tprolongate [{i+1}]=>[{i}]")
            dx = grids[i].prolongate(solns[i+1])
            df = mats[i].dot(dx)

            # if debug_print: 
            #     # # try single element interpolations first..
            #     import matplotlib.pyplot as plt
            #     # ielem_c = 0
            #     # ielem_c = 1
            #     ielem_c = 2
            #     elem_u_c = solns[i+1][2*ielem_c : (2 * ielem_c + 4)]
            #     from src._eb_elem import interp_hermite_disp, interp_lagrange_rotation
            #     xi = np.linspace(-1.0, 1.0, 5)
            #     he = grids[1].xscale
            #     print(f"{elem_u_c=}")

            #     w = np.array([interp_hermite_disp(_xi, elem_u_c, he) for _xi in xi])
            #     th = np.array([interp_lagrange_rotation(_xi, elem_u_c) for _xi in xi])

            #     fig, ax = plt.subplots(1, 2, figsize=(12, 9))
            #     ax[0].plot(xi, w, 'o-')
            #     ax[1].plot(xi, th, 'o-')
            #     plt.show()

            # # temp debug, compare prolong to exact fine solution here
            # if debug_print:
            #     import matplotlib.pyplot as plt
            #     fine_soln = sp.sparse.linalg.spsolve(mats[i].copy(), defects[i])
            #     fig, ax = plt.subplots(2, 2, figsize=(12, 9))
            #     vpn = dof_per_node
            #     ax[0,0].plot(grids[0].xvec, dx[0::vpn], 'o-')
            #     ax[0,1].plot(grids[0].xvec, fine_soln[0::vpn], 'o-')
            #     idof = 1
            #     # idof = 2
            #     ax[1,0].plot(grids[0].xvec, dx[idof::vpn])
            #     ax[1,1].plot(grids[0].xvec, fine_soln[idof::vpn])
            #     plt.show()

            # defect_init = defects[i].copy()

            # temp debug, comparedf to current defect
            if debug_print:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 2, figsize=(12, 9))
                vpn = dof_per_node
                ax[0,0].plot(grids[0].xvec, df[0::vpn])
                ax[0,1].plot(grids[0].xvec, defects[i][0::vpn])
                idof = 1
                # idof = 2
                ax[1,0].plot(grids[0].xvec, df[idof::vpn])
                ax[1,1].plot(grids[0].xvec, defects[i][idof::vpn])
                plt.show()

            # line search scaling of prolongation (since coarse grid less nodes, one DOF scaling not appropriate on default, 
            # can be off by 2x, 4x or some other constant usually)
            omega = np.dot(dx, defects[i]) / np.dot(dx, df)
            
            solns[i] += omega * dx
            defects[i] -= omega * df
            if debug_print: print(f"\tprolong line search with {omega=:.2e}")

            # post-smooth
            if debug_print: print(f"\tpost-smooth grid[{i}]")
            solns[i], defects[i] = block_gauss_seidel_smoother(mats[i], solns[i], defects[i], num_iter=post_smooth, dof_per_node=dof_per_node)

        # check conv
        defect_norm = np.linalg.norm(defects[0])
        print(f"V[{i_cycle}] : {defect_norm=:.2e}")
        if defect_norm <= 1e-6 * init_defect_norm:
            converged = True
            break

    converged_str = "converged" if converged else "didn't converge"
    print(f"V-cycle multigrid {converged_str} in {i_cycle} steps")

    # check the residual on fine grid after this solve
    fine_resid = np.linalg.norm(grids[0].force.copy() - mats[0].dot(solns[0]))
    print(f"\tcheck : {fine_resid=:.2e}")

    return solns[0], i_cycle+1 # return fine grid solution
