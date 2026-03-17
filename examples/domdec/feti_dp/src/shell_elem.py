import numpy as np
import scipy.sparse as sp

from .basis import second_order_quadrature, first_order_quadrature
from .basis import get_lagrange_basis_2d_all

import numpy as np
import scipy.sparse as sp

class MITCShellElement:
    """
    Q4 Reissner–Mindlin shell element for linear structure.
    Uses standard prolongator.

    Adapted from / extension of adv_elems/2_plate/src/elem/mitc_elem.py
    """

    K_SHEAR = 5.0 / 6.0
    DRILL_REG = 10.0
    # DRILL_REG = 0.0 # temp debug

    def __init__(
        self,
        prolong_mode: str = "locking-global",  # 'standard'
        lam: float = 1e-5,
        n_lock_sweeps:int=10,
        omega:float=0.5,
        debug:bool=False,
        strain_debug_mode:str='all', # 'all' is not debug, 'tying', 'bending', 'drill' just does one term
    ):
        assert prolong_mode in ["locking-global", "locking-local", "standard", "energy-jacobi"]

        self.dof_per_node = 6
        self.nodes_per_elem = 4
        self.ndof = self.dof_per_node * self.nodes_per_elem
        self.debug = debug
        self.strain_debug_mode = strain_debug_mode

        self.prolong_mode = prolong_mode
        self.clamped = False
        self.lam = float(lam)
        self.n_lock_sweeps = n_lock_sweeps
        self.omega = float(omega)

        # cache for prolong/restrict operators
        self._P1_cache = {}   # key: nxe_coarse -> P1 (csr)
        self._P2_cache = {}   # key: nxe_coarse -> P2 (csr)
        self._P2_u3_cache = {}
        # self._lock_P_cache = {}
        self._kmat_cache = {}
        self._xpts_cache = {}
        self._P_cache = {}

    # ----------------------------
    # Element stiffness (MITC4)
    # ----------------------------
    def get_kelem(self, E: float, nu: float, thick: float, elem_xpts: np.ndarray):
        """
        Method based on k_add_jacobian_fast assembly from MITC shell GPU code
        """
        pts, wts = first_order_quadrature()

        kelem = np.zeros((self.ndof, self.ndof))

        comp_data = (E, nu, thick)
        fn = self._get_shell_normals(elem_xpts)

        for iquad in range(4):
            
            # get quad pt
            ixi, ieta = iquad % 2, iquad // 2
            xi = pts[ixi]; eta = pts[ieta]
            wt = wts[ixi] * wts[ieta]
            pt = [xi, eta]

            # print(f"{pt=}")
            # get xpts related data
            detXd = self._get_detXd(pt, elem_xpts, fn)
            scale = detXd * wt
            XdinvT, Tmat, XdinvzT = self._get_shell_rotations(
                pt, elem_xpts, fn
            )

            # now loop over each of 24 DOF elem col
            for ideriv in range(self.ndof):
                p_vars = np.zeros(self.ndof)
                p_vars[ideriv] = 1.0
                local_mat_col = np.zeros(self.ndof)

                if self.strain_debug_mode in ['all', 'drill']:
                    # old mode (not fast version)
                    # local_mat_col += self._add_jac_col_drill_nodal(
                    #     pt, scale, elem_xpts, fn,
                    #     XdinvT, Tmat, XdinvzT, comp_data,
                    #     p_vars
                    # )

                    # checked it's fine to use either version of drill strain and it still gives mesh convergence!
                    # just wanted to be sure
                    local_mat_col += self._add_jac_col_drill(
                        pt, scale, elem_xpts, fn,
                        XdinvT, Tmat, XdinvzT, comp_data,
                        p_vars
                    )

                if self.strain_debug_mode in ['all', 'bending']:
                    local_mat_col += self._add_jac_col_bending(
                        pt, scale, elem_xpts, fn,
                        XdinvT, Tmat, XdinvzT, comp_data,
                        p_vars
                    )

                if self.strain_debug_mode in ['all', 'tying']:
                    local_mat_col += self._add_jac_col_tying(
                        pt, scale, elem_xpts, fn,
                        XdinvT, Tmat, XdinvzT, comp_data,
                        p_vars
                    )

                # add mat col into the kelem
                kelem[:, ideriv] += local_mat_col

        # import matplotlib.pyplot as plt
        # plt.imshow(kelem)
        # plt.show()

        return kelem
    
    def _interp_fields(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        out = np.zeros(num_fields)
        N, _, _ = get_lagrange_basis_2d_all(pt[0], pt[1])
        # print(f"{pt=} {N=}")
        for ifield in range(num_fields):
            for inode in range(4):
                out[ifield] += N[inode] * vec[vpn * inode + ifield]
        return out
    
    def _interp_fields_transpose(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        out = np.zeros(vpn * 4)
        N, _, _ = get_lagrange_basis_2d_all(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                out[vpn * inode + ifield] += N[inode] * vec[ifield]
        return out

    def _interp_fields_grad(self, pt:list, vpn:int, num_fields:int, vec:np.ndarray):
        dxi = np.zeros(num_fields); deta = np.zeros(num_fields)
        
        _, dNdxi, dNdeta = get_lagrange_basis_2d_all(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                dxi[ifield] += dNdxi[inode] * vec[vpn * inode + ifield]
                deta[ifield] += dNdeta[inode] * vec[vpn * inode + ifield]
        return dxi, deta
    
    def _interp_fields_grad_transpose(self, pt:list, vpn:int, num_fields:int, dxi:np.ndarray, deta:np.ndarray):
        dout = np.zeros(vpn * 4)

        _, dNdxi, dNdeta = get_lagrange_basis_2d_all(pt[0], pt[1])
        for ifield in range(num_fields):
            for inode in range(4):
                dout[vpn * inode + ifield] += dxi[ifield] * dNdxi[inode] + \
                    deta[ifield] * dNdeta[inode]
        return dout

    def _get_shell_normals(self, elem_xpts:np.ndarray):
        fn = np.zeros(12)

        # print(f"{elem_xpts.reshape((4,3))=}")

        for i in range(4):
            # node pt
            ixi, ieta = i % 2, i // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            # print(f"{dX_dxi=} {dX_deta=}")
            normal = np.cross(dX_dxi, dX_deta)
            normal /= np.linalg.norm(normal)
            fn[3*i:(3*i+3)] += normal

            # xpt = self._interp_fields(node_pt, vpn=3, num_fields=3, vec=elem_xpts)
            # print(f"{dX_dxi=} {dX_deta=} {normal=}")

        # fn_rs = fn.reshape((3,4))
        # print(f"{fn_rs=}")
        return fn
    
    def _get_detXd(self, pt:list, elem_xpts:np.ndarray, fn:np.ndarray):
        n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
        dX_dxi, dX_deta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)        
        Xd = np.column_stack((dX_dxi, dX_deta, n0))
        _detXd = np.linalg.det(Xd)
        return np.abs(_detXd)
    
    def _shell_compute_transform(
        self, 
        n0:np.ndarray, 
        ref_axis:np.ndarray=np.array([1, 0, 0]),
    ):
        
        nhat = n0 / np.linalg.norm(n0)
        t1 = ref_axis
        t1hat = t1 - np.dot(t1, nhat) * nhat
        t1hat /= np.linalg.norm(t1hat)
        t2hat = np.cross(nhat, t1hat)
        Tmat = np.column_stack((t1hat, t2hat, nhat))
        # should just be eye(3) or rotation mat version of that often times
        # but not on cylindrical shell, then changes some depending on curvilinear spot
        # print(f"{Tmat=}")
        return Tmat
    
    def _get_shell_rotations(self, pt, elem_xpts, fn):
        # compute Tmat, XdinvT, XdinvzT transformation matrices for shell rotation
        n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
        dX_dxi, dX_deta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)        
        nxi, neta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=fn)
        Xd = np.column_stack((dX_dxi, dX_deta, n0))
        Xdz = np.column_stack((nxi, neta, np.zeros(3)))
        Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))

        # print(f"{Xd=}")

        Xdinv = np.linalg.inv(Xd)
        XdinvT = Xdinv @ Tmat
        XdinvzT = -Xdinv @ Xdz @ XdinvT

        # print(f"{XdinvT=} {Tmat=} {XdinvzT=}")
        return XdinvT, Tmat, XdinvzT
    
    def _add_jac_col_drill_nodal(
        self, quad_pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        _XdinvT:np.ndarray, _Tmat:np.ndarray, 
        _XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray   
    ):
        mat_col = np.zeros(24)

        etn = np.zeros(4)
        for inode in range(4):
            # node pt
            ixi, ieta = inode % 2, inode // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)        
            n0 = fn[3*inode:(3*inode+3)]
            Xd = np.column_stack((dX_dxi, dX_deta, n0))
            Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))
            u0xi, u0eta = self._interp_fields_grad(node_pt, vpn=6, num_fields=3, vec=p_vars)
            u0xn = np.column_stack((u0xi, u0eta, np.zeros(3)))
            u0 = p_vars[6*inode:(6*inode+6)]
            thx, thy, thz = u0[3], u0[4], u0[5]
            C0 = np.array([ # rotation matrix for drill strain
                [1.0, thz, -thy],
                [-thz, 1.0, thx],
                [thy, -thx, 1.0]
            ])
            C = Tmat.T @ C0 @ Tmat
            XdinvT = np.linalg.inv(Xd) @ Tmat
            u0x = Tmat.T @ u0xn @ XdinvT
            # eval drill strain
            etn[inode] = 0.5 * (C[1,0] + u0x[1,0] - C[0,1] - u0x[0,1])
    
        et = self._interp_fields(quad_pt, vpn=1, num_fields=1, vec=etn)
        
        # now compute drill stress
        # st = self._compute_drill_stress(scale, comp_data, et)
        E, nu, thick = comp_data
        G = E / 2.0 / (1.0 + nu)
        As = self.K_SHEAR * G * thick
        A_drill = As * self.DRILL_REG
        st = scale * A_drill * et[0] # equiv to et_bar

        # reverse
        st_vec = np.zeros(1); st_vec[0] = st
        stn = self._interp_fields_transpose(quad_pt, vpn=1, num_fields=1, vec=st_vec)

        for inode in range(4):
            # node pt
            ixi, ieta = inode % 2, inode // 2
            xi = -1 + 2.0 * ixi
            eta = -1 + 2.0 * ieta
            node_pt = [xi, eta]
            # print(f"{elem_xpts=}")

            dX_dxi, dX_deta = self._interp_fields_grad(node_pt, vpn=3, num_fields=3, vec=elem_xpts)        
            n0 = fn[3*inode:(3*inode+3)]
            Xd = np.column_stack((dX_dxi, dX_deta, n0))
            Tmat = self._shell_compute_transform(n0, ref_axis=np.array([1, 0, 0]))
            XdinvT = np.linalg.inv(Xd) @ Tmat            
            
            # now do the transpose version of the drill strains
            # mat_col = self._compute_drill_strain_transpose(pt, Tmat, XdinvT, st)
            u0x_hat = np.zeros_like(u0x); C_hat = np.zeros_like(C)
            u0x_hat[1,0] += 0.5 * stn[inode]
            C_hat[1,0] += 0.5 * stn[inode]
            u0x_hat[0,1] -= 0.5 * stn[inode]
            C_hat[0,1] -= 0.5 * stn[inode]

            C0_hat = Tmat @ C_hat @ Tmat.T
            u0xn_hat = Tmat @ u0x_hat @ XdinvT.T
            u0xi_hat = u0xn_hat[:,0]
            u0eta_hat = u0xn_hat[:,1]

            thx_hat = C0_hat[1,2] - C0_hat[2,1]
            thy_hat = C0_hat[2,0] - C0_hat[0,2]
            thz_hat = C0_hat[0,1] - C0_hat[1,0]

            mat_col[6*inode+3] += thx_hat
            mat_col[6*inode+4] += thy_hat
            mat_col[6*inode+5] += thz_hat
            
            mat_col += self._interp_fields_grad_transpose(
                node_pt, vpn=6, num_fields=3, dxi=u0xi_hat, deta=u0eta_hat
            )

        return mat_col
        
    def _add_jac_col_drill(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # just the linear drill strains here (so no nonlinearity)
        # et = self._compute_drill_strain(pt, Tmat, XdinvT, p_vars)
        # done explicitly here
        u0xi, u0eta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        u0xn = np.column_stack((u0xi, u0eta, np.zeros(3)))
        u0 = self._interp_fields(pt, vpn=6, num_fields=6, vec=p_vars)
        thx, thy, thz = u0[3], u0[4], u0[5]
        C0 = np.array([ # rotation matrix for drill strain
            [1.0, thz, -thy],
            [-thz, 1.0, thx],
            [thy, -thx, 1.0]
        ])
        C = Tmat.T @ C0 @ Tmat
        u0x = Tmat.T @ u0xn @ XdinvT
        # eval drill strain
        et = 0.5 * (C[1,0] + u0x[1,0] - C[0,1] - u0x[0,1])

        # now compute drill stress
        # st = self._compute_drill_stress(scale, comp_data, et)
        E, nu, thick = comp_data
        G = E / 2.0 / (1.0 + nu)
        As = self.K_SHEAR * G * thick
        A_drill = As * self.DRILL_REG
        st = scale * A_drill * et # equiv to et_bar

        # now do the transpose version of the drill strains
        # mat_col = self._compute_drill_strain_transpose(pt, Tmat, XdinvT, st)
        u0x_hat = np.zeros_like(u0x); C_hat = np.zeros_like(C)
        u0x_hat[1,0] += 0.5 * st
        C_hat[1,0] += 0.5 * st
        u0x_hat[0,1] -= 0.5 * st
        C_hat[0,1] -= 0.5 * st

        C0_hat = Tmat @ C_hat @ Tmat.T
        u0xn_hat = Tmat @ u0x_hat @ XdinvT.T
        u0xi_hat = u0xn_hat[:,0]
        u0eta_hat = u0xn_hat[:,1]

        thx_hat = C0_hat[1,2] - C0_hat[2,1]
        thy_hat = C0_hat[2,0] - C0_hat[0,2]
        thz_hat = C0_hat[0,1] - C0_hat[1,0]

        u0_sens = np.zeros(6)
        u0_sens[3] += thx_hat
        u0_sens[4] += thy_hat
        u0_sens[5] += thz_hat

        mat_col = self._interp_fields_transpose(
            pt, vpn=6, num_fields=6, vec=u0_sens
        )
        mat_col += self._interp_fields_grad_transpose(
            pt, vpn=6, num_fields=3, dxi=u0xi_hat, deta=u0eta_hat
        )
        
        return mat_col
    
    def _compute_director(self, p_vars, fn):
        d = np.zeros(12)
        for i in range(4):
            d[3*i:(3*i+3)] += np.cross(
                p_vars[(6*i+3):(6*i+6)],
                fn[3*i:(3*i+3)],
            )
        return d
    
    def _compute_director_sens(self, fn, d_bar):
        mat_col = np.zeros(24)
        for i in range(4):
            mat_col[(6*i+3):(6*i+6)] += np.cross(
                fn[3*i:(3*i+3)],
                d_bar[3*i:(3*i+3)],
            )
        return mat_col
    
    def _compute_mitc_tying_strains(self, elem_xpts, fn, p_vars, p_d):
        # compute each of the tying strains (linear form only here)

        # 9 total tying strains in combinations of 2 and 1 per each of 5 strains
        # order is [2 x g11, 2 x g22, 1 x g12, 2 x g23, 2 x g13]
        ety = np.zeros(9) 

        # g11 tying strain
        for j in range(2):
            # tying pt
            pt = [0.0, -1 + 2 * j]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            ety[j] = np.dot(Uxi, Xxi)

        # g22 tying strain
        for j in range(2):
            # tying pt
            pt = [-1 + 2 * j, 0.0]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            ety[2 + j] = np.dot(Ueta, Xeta)

        # g12 tying strain
        pt = [0.0, 0.0]
        Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
        Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        ety[4] = 0.5 * (np.dot(Uxi, Xeta) + np.dot(Ueta, Xxi))

        # g23 tying strain
        for j in range(2):
            # tying pt
            pt = [-1 + 2 * j, 0.0]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
            ety[5 + j] = 0.5 * (np.dot(Xeta, d0) + np.dot(n0, Ueta))

        # g13 tying strain
        for j in range(2):
            # tying pt
            pt = [0.0, -1 + 2 * j]
            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
            ety[7 + j] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Uxi))

        return ety
    
    def _compute_mitc_tying_strains_transpose(self, elem_xpts, fn, p_vars, p_d, ety_bar):
        """
        Transpose/adjoint of _compute_mitc_tying_strains.

        Inputs
        ------
        elem_xpts : geometry nodal vector (used in forward for Xxi/Xeta)
        fn        : nodal director/normal vector (used in forward for n0)
        p_vars    : nodal dofs (num_fields=6)
        p_d       : nodal director-like dofs (num_fields=3)
        ety_bar   : adjoint of ety, shape (9,)

        Returns
        -------
        mat_col : sensitivity w.r.t. p_vars (same shape as p_vars)
        d_bar   : sensitivity w.r.t. p_d   (same shape as p_d)
        """
        # ety_bar = np.asarray(ety_bar, dtype=float).reshape(9,)

        mat_col = np.zeros_like(p_vars, dtype=float)
        d_bar   = np.zeros_like(p_d, dtype=float)

        # -------------------------
        # g11 tying strain (2 pts): ety[j] = dot(Uxi, Xxi)
        # -------------------------
        for j in range(2):
            pt = [0.0, -1.0 + 2.0 * j]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

            s = float(ety_bar[j])

            # ety = dot(Uxi, Xxi)
            Uxi_bar  = s * Xxi
            Ueta_bar = np.zeros_like(Ueta)

            # push back to p_vars via grad-transpose
            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )

        # -------------------------
        # g22 tying strain (2 pts): ety[2+j] = dot(Ueta, Xeta)
        # -------------------------
        for j in range(2):
            pt = [-1.0 + 2.0 * j, 0.0]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

            s = float(ety_bar[2 + j])

            Uxi_bar  = np.zeros_like(Uxi)
            Ueta_bar = s * Xeta

            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )

        # -------------------------
        # g12 tying strain (center):
        # ety[4] = 0.5*(dot(Uxi,Xeta) + dot(Ueta,Xxi))
        # -------------------------
        pt = [0.0, 0.0]
        Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
        Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)

        s = float(ety_bar[4])

        Uxi_bar  = 0.5 * s * Xeta
        Ueta_bar = 0.5 * s * Xxi

        mat_col += self._interp_fields_grad_transpose(
            pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
        )

        # -------------------------
        # g23 tying strain (2 pts):
        # ety[5+j] = 0.5*(dot(Xeta,d0) + dot(n0,Ueta))
        # -------------------------
        for j in range(2):
            pt = [-1.0 + 2.0 * j, 0.0]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)

            s = float(ety_bar[5 + j])

            # ety = 0.5*(dot(Xeta,d0) + dot(n0,Ueta))
            # d0_bar from dot(Xeta,d0)
            d0_bar = 0.5 * s * Xeta

            # Ueta_bar from dot(n0,Ueta)
            Uxi_bar  = np.zeros_like(Uxi)
            Ueta_bar = 0.5 * s * n0

            # push back to p_vars and p_d
            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )
            d_bar += self._interp_fields_transpose(
                pt, vpn=3, num_fields=3, vec=d0_bar
            )

        # -------------------------
        # g13 tying strain (2 pts):
        # ety[7+j] = 0.5*(dot(Xxi,d0) + dot(n0,Uxi))
        # -------------------------
        for j in range(2):
            pt = [0.0, -1.0 + 2.0 * j]

            Xxi, Xeta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=elem_xpts)
            Uxi, Ueta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)
            d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)

            s = float(ety_bar[7 + j])

            # ety = 0.5*(dot(Xxi,d0) + dot(n0,Uxi))
            d0_bar = 0.5 * s * Xxi

            Uxi_bar  = 0.5 * s * n0
            Ueta_bar = np.zeros_like(Ueta)

            mat_col += self._interp_fields_grad_transpose(
                pt, vpn=6, num_fields=3, dxi=Uxi_bar, deta=Ueta_bar
            )
            d_bar += self._interp_fields_transpose(
                pt, vpn=3, num_fields=3, vec=d0_bar
            )

        return mat_col, d_bar
    
    def _1d_tying_interp(self, u, vec: np.ndarray):
        N0 = 0.5 * (1.0 - u)
        N1 = 0.5 * (1.0 + u)
        return N0 * vec[0] + N1 * vec[1]

    def _1d_tying_interp_transpose(self, u, out_bar: float):
        """
        Transpose of: out = N0*v0 + N1*v1
        Returns contributions to [v0_bar, v1_bar].
        """
        N0 = 0.5 * (1.0 - u)
        N1 = 0.5 * (1.0 + u)
        return np.array([N0 * out_bar, N1 * out_bar], dtype=float)

    def _interp_tying_strains(self, quad_pt, p_ety):
        xi = quad_pt[0]; eta = quad_pt[1]
        gty = np.zeros((3, 3))

        # membrane normals
        gty[0, 0] = self._1d_tying_interp(u=eta, vec=[p_ety[0], p_ety[1]]) # g11
        gty[1, 1] = self._1d_tying_interp(u=xi,  vec=[p_ety[2], p_ety[3]]) # g22

        # in-plane shear (symmetric)
        gty[0, 1] = gty[1, 0] = p_ety[4] # g12

        # transverse shear terms (symmetric)
        val12 = self._1d_tying_interp(u=xi,  vec=[p_ety[5], p_ety[6]])
        gty[1, 2] = gty[2, 1] = val12 # g23

        val02 = self._1d_tying_interp(u=eta, vec=[p_ety[7], p_ety[8]])
        gty[0, 2] = gty[2, 0] = val02 # g12

        return gty
    
    def _interp_tying_strains_transpose(self, quad_pt, p_gty):
        xi = quad_pt[0]; eta = quad_pt[1]
        p_ety = np.zeros(9, dtype=float)

        # gty[0,0]
        p_ety[0:2] += self._1d_tying_interp_transpose(
            u=eta, out_bar=float(p_gty[0, 0])
        )

        # gty[1,1]
        p_ety[2:4] += self._1d_tying_interp_transpose(
            u=xi, out_bar=float(p_gty[1, 1])
        )

        # gty[0,1] = gty[1,0] = p4
        p_ety[4] += float(p_gty[0, 1]) + float(p_gty[1, 0])

        # gty[1,2] = gty[2,1]
        shear12_bar = float(p_gty[1, 2]) + float(p_gty[2, 1])
        p_ety[5:7] += self._1d_tying_interp_transpose(
            u=xi, out_bar=shear12_bar
        )

        # gty[0,2] = gty[2,0]
        shear02_bar = float(p_gty[0, 2]) + float(p_gty[2, 0])
        p_ety[7:9] += self._1d_tying_interp_transpose(
            u=eta, out_bar=shear02_bar
        )

        return p_ety
    
    def _get_tying_strain_matrix(
        self, quad_pt:list,
        elem_xpts:np.ndarray
    ):
        # for multigrid operators
        tying_matrix = np.zeros((5, 24))

        fn = self._get_shell_normals(elem_xpts)
        XdinvT, _, _ = self._get_shell_rotations(
            quad_pt, elem_xpts, fn
        )
        
        for i in range(24):
            p_vars = np.zeros(24)
            p_vars[i] = 1.0

            # just do pforward and hrev part cause linear
            p_d = self._compute_director(p_vars, fn)
            p_ety = self._compute_mitc_tying_strains(elem_xpts, fn, p_vars, p_d)
            p_gty = self._interp_tying_strains(quad_pt, p_ety)
            # sym mat rotate frame
            p_e0ty = XdinvT.T @ p_gty @ XdinvT
            # membrane strains
            p_em = np.array([p_e0ty[0,0], p_e0ty[1,1], 2.0 * p_e0ty[0,1]]) 
            # trv shear strains 
            p_es = 2.0 * np.array([p_e0ty[0,2], p_e0ty[1,2]]) # or just sym sums here which means reverse is copy to each component

            p_tying = np.concatenate([p_em, p_es], axis=0)
            tying_matrix[:,i] = p_tying

        return tying_matrix # return the five tying strains at this quadpt

    def _add_jac_col_tying(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # just do pforward and hrev part cause linear
        p_d = self._compute_director(p_vars, fn)
        p_ety = self._compute_mitc_tying_strains(elem_xpts, fn, p_vars, p_d)
        p_gty = self._interp_tying_strains(pt, p_ety)
        # sym mat rotate frame
        p_e0ty = XdinvT.T @ p_gty @ XdinvT
        # membrane strains
        p_em = np.array([p_e0ty[0,0], p_e0ty[1,1], 2.0 * p_e0ty[0,1]]) 
        # trv shear strains 
        p_es = 2.0 * np.array([p_e0ty[0,2], p_e0ty[1,2]]) # or just sym sums here which means reverse is copy to each component

        # tangent stiffness 2D matrix
        E, nu, thick = comp_data
        C = E / (1.0 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1.0 - nu) / 2.0]
        ])
        # print(f"{E=} {nu=} {thick=}")
        A = C * thick * scale # A matrix from shell theory
        # B matrix is zero
        # trv shear A matrix
        ks = 5.0 / 6.0
        As = np.eye(2) * A[-1,-1] * ks

        # membrane and trv shear stresses
        h_sm = np.dot(A, p_em)
        h_ss = np.dot(As, p_es)

        # forward is double that or you can just add both off-diag entries
        # so reverse / transpose is still copy to each off-diag component not double
        h_e0ty = np.array([
            [h_sm[0], h_sm[2], h_ss[0]],
            [h_sm[2], h_sm[1], h_ss[1]],
            [h_ss[0], h_ss[1], 0.0],
        ])

        h_gty = XdinvT @ h_e0ty @ XdinvT.T
        h_ety = self._interp_tying_strains_transpose(pt, h_gty)
        add_mat_col, d_hat = self._compute_mitc_tying_strains_transpose(elem_xpts, fn, p_vars, p_d, h_ety)
        mat_col = add_mat_col
        mat_col += self._compute_director_sens(fn, d_hat)
        return mat_col
    
    def _compute_bending_disp_grad(self, pt, p_vars, p_d, Tmat, XdinvT, XdinvzT):    
        d0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=p_d)
        d0xi, d0eta = self._interp_fields_grad(pt, vpn=3, num_fields=3, vec=p_d)
        u0xi, u0eta = self._interp_fields_grad(pt, vpn=6, num_fields=3, vec=p_vars)
        u0x_init = np.column_stack((u0xi, u0eta, d0))
        u1x_init = np.column_stack((d0xi, d0eta, np.zeros(3)))

        p_u1x = Tmat.T @ (u1x_init @ XdinvT + u0x_init @ XdinvzT)
        p_u0x = Tmat.T @ u0x_init @ XdinvT
        return p_u0x, p_u1x

    def _compute_bending_strain(self, p_u0x, p_u1x):
        # linear strains only here
        p_ek = np.array([
            p_u1x[0,0],
            p_u1x[1,1],
            p_u1x[0,1] + p_u1x[1,0],
        ])
        return p_ek

    def _compute_bending_strain_sens(self, h_ek):
        h_u0x = np.zeros((3,3))
        h_u1x = np.zeros((3,3))
        h_u1x[0,0] += h_ek[0]
        h_u1x[1,1] += h_ek[1]
        h_u1x[0,1] += h_ek[2]
        h_u1x[1,0] += h_ek[2]
        return h_u0x, h_u1x

    def _compute_bending_disp_grad_sens(self, pt, Tmat, XdinvT, XdinvzT, h_u0x, h_u1x):
        
        u0d_barT = XdinvT @ h_u0x.T @ Tmat.T
        u0d_barT += XdinvzT @ h_u1x.T @ Tmat.T
        h_d = self._interp_fields_transpose(pt, vpn=3, num_fields=3, vec=u0d_barT[2,:])
        mat_col = self._interp_fields_grad_transpose(pt, vpn=6, num_fields=3, dxi=u0d_barT[0,:], deta=u0d_barT[1,:])

        u1d_barT = XdinvT @ h_u1x.T @ Tmat.T
        h_d += self._interp_fields_grad_transpose(pt, vpn=3, num_fields=3, dxi=u1d_barT[0,:], deta=u1d_barT[1,:])
        return mat_col, h_d

    def _add_jac_col_bending(
        self, pt:list, scale:float, 
        elem_xpts:np.ndarray, fn:np.ndarray,
        XdinvT:np.ndarray, Tmat:np.ndarray, 
        XdinvzT:np.ndarray, comp_data:list,
        p_vars:np.ndarray
    ):

        # p-forward section
        p_d = self._compute_director(p_vars, fn)
        p_u0x, p_u1x = self._compute_bending_disp_grad(pt, p_vars, p_d, Tmat, XdinvT, XdinvzT)
        p_ek = self._compute_bending_strain(p_u0x, p_u1x)

        # compute bending stress (constitutive)
        E, nu, thick = comp_data
        C = E / (1.0 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1.0 - nu) / 2.0]
        ])
        D = C * thick**3 / 12.0 * scale # A matrix from shell theory
        h_ek = np.dot(D, p_ek)

        # h-reverse section
        h_u0x, h_u1x = self._compute_bending_strain_sens(h_ek)
        add_mat_col, h_d = self._compute_bending_disp_grad_sens(pt, Tmat, XdinvT, XdinvzT, h_u0x, h_u1x)
        mat_col = add_mat_col
        mat_col += self._compute_director_sens(fn, h_d)
        return mat_col

    def get_felem(self, mag, elem_xpts: np.ndarray):
        pts, wts = first_order_quadrature()
        felem = np.zeros(self.ndof)

        fn = self._get_shell_normals(elem_xpts)  # once

        for iquad in range(4):
            ixi, ieta = iquad % 2, iquad // 2
            xi = pts[ixi]; eta = pts[ieta]
            wt = wts[ixi] * wts[ieta]
            pt = [xi, eta]

            detXd = self._get_detXd(pt, elem_xpts, fn)
            scale = abs(detXd) * wt   # consider abs

            N, _, _ = get_lagrange_basis_2d_all(xi, eta)

            xyzq = self._interp_fields(pt, vpn=3, num_fields=3, vec=elem_xpts)
            q = float(mag(xyzq[0], xyzq[1], xyzq[2]))

            n0 = self._interp_fields(pt, vpn=3, num_fields=3, vec=fn)

            # add q*n0 to translational DOFs (u,v,w) at each node
            felem[0::6] += n0[0] * (scale * q) * N
            felem[1::6] += n0[1] * (scale * q) * N
            felem[2::6] += n0[2] * (scale * q) * N

        return felem