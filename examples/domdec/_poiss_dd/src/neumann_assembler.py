    
from .assembler import SubdomainPlateAssembler
import numpy as np
import scipy.sparse as sp

class NeumannPlateAssembler(SubdomainPlateAssembler):
    def __init__(
        self,
        ELEMENT,
        nxe: int,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        nxs:int=1, # num subdomains in x-direc
        nys:int=1, # num subdomains in y-direc
    ):
        # call main init, including nz pattern for subdomains (but no block structure)
        SubdomainPlateAssembler.__init__(self, ELEMENT, nxe, length, width, load_fcn, clamped, nxs, nys)

    def _nodes_to_dof(self, node_arr:np.ndarray):
        dpn = self.dof_per_node
        return np.array([dpn*inode + idof for inode in node_arr for idof in range(dpn)])

    def get_interface_rhs(self):
        global_out = np.zeros(self.N, dtype=float)

        for i_subdomain in range(self.num_subdomains):
            sd_nodes = self.sd_nodes[i_subdomain]
            sd_dof = self._nodes_to_dof(sd_nodes)

            sd_rhs = self.sd_force[i_subdomain]   # <-- key fix

            interior_dof = self.sd_interior_dofs[i_subdomain]
            interface_dof = self.sd_interface_dofs[i_subdomain]

            f_I = sd_rhs[interior_dof]
            f_G = sd_rhs[interface_dof]

            temp_I = sp.linalg.spsolve(self.sd_A_II[i_subdomain].tocsc(), f_I)
            out_G = f_G - self.sd_A_GI[i_subdomain].dot(temp_I)

            sd_out = np.zeros_like(sd_rhs)
            sd_out[interface_dof] = out_G
            global_out[sd_dof] += sd_out

        interface_out = global_out[self.interface_dof_global]
        return interface_out

    def precond_solve(self, interface_rhs:np.ndarray):
        # neumann-neumann preconditioner (multi-subdomain)

        # solves on interface unknowns
        # short-cut right now (get interface DOF like this)
        global_rhs = np.zeros(self.N)
        global_rhs[self.interface_dof_global] = interface_rhs.copy()
        global_soln = np.zeros_like(global_rhs)
        # TODO : in CUDA GPU code should avoid operations at global size unless just copying
        # stuff which is fine I guess.. (just don't do mat-vec etc at global size)

        for i_subdomain in range(self.num_subdomains):
            # get the nodes in this subdomain
            sd_nodes = self.sd_nodes[i_subdomain]
            sd_dof = self._nodes_to_dof(sd_nodes)
            sd_rhs = global_rhs[sd_dof] # with zero on interior nodes automatically

            # solve the subdomain problem
            sd_soln = sp.linalg.spsolve(self.sd_kmat[i_subdomain].tocsc(), sd_rhs)

            # add back to global
            global_soln[sd_dof] += sd_soln

        # now extract only interface part of soln (thus ignoring interior parts)
        interface_soln = global_soln[self.interface_dof_global]
        return interface_soln

    def mat_vec(self, interface_disp:np.ndarray):
        # compute interface schur-complement products by interior local solves

        global_disp = np.zeros(self.N)
        global_disp[self.interface_dof_global] = interface_disp.copy()
        global_out = np.zeros(self.N)
        for i_subdomain in range(self.num_subdomains):
            # get the nodes in this subdomain
            sd_nodes = self.sd_nodes[i_subdomain]
            sd_dof = self._nodes_to_dof(sd_nodes)
            interface_dof = self.sd_interface_dofs[i_subdomain]
            sd_disp = global_disp[sd_dof] # with zero DOF on interior obviously
            u_G = sd_disp[interface_dof]
            # u_I = sd_disp[self.sd_interior_dofs] # which is zero

            # now perform Schur-complement mat-vec operation
            # (A_GG - A_GI * A_II^{-1} * A_IG) * u_G for subdomain
            out_G = self.sd_A_GG[i_subdomain].dot(u_G)
            temp_I1 = self.sd_A_IG[i_subdomain].dot(u_G)
            temp_I2 = sp.linalg.spsolve(self.sd_A_II[i_subdomain].tocsc(), temp_I1)
            temp_G = self.sd_A_GI[i_subdomain].dot(temp_I2)
            out_G -= temp_G
            sd_out = np.zeros_like(sd_disp)
            sd_out[interface_dof] = out_G

            # add to global
            global_out[sd_dof] += sd_out

        # now extract only interface part of global out
        interface_out = global_out[self.interface_dof_global]
        return interface_out
    
    def get_global_solution(self, interface_soln:np.ndarray):
        # after solving reduced interface problem, recover global solution
        # by local solves

        global_soln = np.zeros(self.N)
        global_soln[self.interface_dof_global] = interface_soln.copy()

        for i_subdomain in range(self.num_subdomains):
            # get the nodes in this subdomain
            sd_nodes = self.sd_nodes[i_subdomain]
            sd_dof = self._nodes_to_dof(sd_nodes)
            interior_dof = self.sd_interior_dofs[i_subdomain]
            interface_dof = self.sd_interface_dofs[i_subdomain]
            sd_rhs = self.sd_force[i_subdomain]
            sd_soln = global_soln[sd_dof] # with zero DOF on interior obviously
            f_I = sd_rhs[interior_dof]
            u_G = sd_soln[interface_dof]
            # but interior DOF are zero at this point
            
            # solve single system A_II * u_I + A_IG * u_G = f_I for u_I unknown
            # only u_I is unknown at this point
            f_I2 = f_I - self.sd_A_IG[i_subdomain].dot(u_G)
            u_I = sp.linalg.spsolve(self.sd_A_II[i_subdomain].tocsc(), f_I2)
            sd_addsoln = np.zeros_like(sd_soln)
            sd_addsoln[interior_dof] = u_I
            # only adding interior DOF to global (don't change interface DOF)

            # add to global
            global_soln[sd_dof] += sd_addsoln

        return global_soln