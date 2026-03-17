// archive steps for debugging BddcSolver

T *h_int_rhs = this->u_I.createHostVec().getPtr();
printf("h_gam_rhs:\n");
for (int ilam = 0; ilam < this->I_nnodes; ilam++) {
    int iglob = this->I_nodes[ilam];
    printf("i_int %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_int_rhs[lam_dof]);
    }
    printf("\n");
}

print(f"{i_sd=}")
for j in range(len(I)//3):
    I_node = self.sd_interior_nodes[i_sd][j]
    glob_node = self.sd_nodes[i_sd][I_node]
    sub_vec = uI[3*j:(3*j+3)]
    print(f"\tinterior soln {j=} {glob_node=} {sub_vec}")

T *h_edge_rhs = this->u_I.createHostVec().getPtr();
        printf("h_edge_rhs:\n");
        for (int ilam = 0; ilam < this->lam_nnodes; ilam++) {
    int iglob = this->lam_nodes[ilam];
    printf("ilam %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_edge_rhs[lam_dof]);
    }
    printf("\n");
        }

T *h_tempV = this->temp_V.createHostVec().getPtr();
        printf("h_Vc_rhs:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_tempV[lam_dof]);
    }
    printf("\n");
        }

        T *h_vecgamV = vec_gam.createHostVec().getPtr();
        printf("h_vecgamV:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_vecgamV[edge_size + lam_dof]);
    }
    printf("\n");
        }

T *h_gam_rhs = gam_rhs.createHostVec().getPtr();
        printf("h_gam_rhs:\n");
        for (int ilam = 0; ilam < ngam; ilam++) {
    int iglob = gam_nodes[ilam];
    printf("igam %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_gam_rhs[lam_dof]);
    }
    printf("\n");
        }

for i in range(nEg//3):
            edge_node = self.edge_interface_nodes_global[i]
            sub_vec = gE[3*i:(3*i+3)]
            print(f"gam edge {i=} {edge_node=} {sub_vec}")

for j in range(nVg//3):
    v_node = self.vertex_nodes_global[j]
    sub_vec = gV[3*j:(3*j+3)]
    print(f"gam vertex {j=} {v_node=} {sub_vec}")

print(f"I in pc{i_sd=}")
I_nodes = self.sd_interior_nodes[i_sd]
for j in range(len(I_nodes)):
    I_node = I_nodes[j]
    glob_node = self.sd_nodes[i_sd][I_node]
    sub_vec = uIi[3*j:(3*j+3)]
    print(f"\tinterior soln {j=} {glob_node=} {sub_vec}")

print(f"E in pc{i_sd=}")
E_nodes = self.sd_edge_nodes[i_sd]
for j in range(len(E_nodes)):
    E_node = E_nodes[j]
    glob_node = self.sd_nodes[i_sd][E_node]
    sub_vec = uEi[3*j:(3*j+3)]
    print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

T *h_gamrhsV = gam_rhs.createHostVec().getPtr();
        printf("h_gamrhsV (in solve):\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_gamrhsV[n_edge * this->block_dim + lam_dof]);
    }
    printf("\n");
        }

T *h_f_V = this->f_V.createHostVec().getPtr();
        printf("h_Fv_rhs:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_f_V[lam_dof]);
    }
    printf("\n");
        }

T *h_IE_rhs = this->u_IE.createHostVec().getPtr();
        printf("h_IE_rhs in pc:\n");
        for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    int iglob = this->IE_nodes[ilam];
    printf("i_IE %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_IE_rhs[lam_dof]);
    }
    printf("\n");
        }

T *h_f_V2 = this->f_V.createHostVec().getPtr();
        printf("h_Fv_rhs 2:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_f_V2[lam_dof]);
    }
    printf("\n");
        }

print(f"I in pc{i_sd=}")
            I_nodes = self.sd_interior_nodes[i_sd]
            for j in range(len(I_nodes)):
                I_node = I_nodes[j]
                glob_node = self.sd_nodes[i_sd][I_node]
                sub_vec = uIi[3*j:(3*j+3)]
                print(f"\tinterior soln {j=} {glob_node=} {sub_vec}")

            print(f"E in pc{i_sd=}")
            E_nodes = self.sd_edge_nodes[i_sd]
            for j in range(len(E_nodes)):
                E_node = E_nodes[j]
                glob_node = self.sd_nodes[i_sd][E_node]
                sub_vec = uEi[3*j:(3*j+3)]
                print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

V_nodes = self.vertex_nodes_global
        print("cV rhs (pre IE-corr)")
        for j in range(len(V_nodes)):
            v_node = self.vertex_nodes_global[j]
            sub_vec = cV[3*j:(3*j+3)]
            print(f"fV vertex {j=} {v_node=} {sub_vec}")

for j in range(nVg//3):
            v_node = self.vertex_nodes_global[j]
            sub_vec = gV[3*j:(3*j+3)]
            print(f"gam vertex {j=} {v_node=} {sub_vec}")

V_nodes = self.vertex_nodes_global
        print("cV rhs")
        for j in range(len(V_nodes)):
            v_node = self.vertex_nodes_global[j]
            sub_vec = cV[3*j:(3*j+3)]
            print(f"fV vertex {j=} {v_node=} {sub_vec}")

        print("uV soln")
        for j in range(len(V_nodes)):
            v_node = self.vertex_nodes_global[j]
            sub_vec = uV[3*j:(3*j+3)]
            print(f"uV vertex {j=} {v_node=} {sub_vec}")


            E_rhs = -self.sd_A_EV[i_sd].dot(uVi)
            print(f"E rhs in pc{i_sd=}")
            E_nodes = self.sd_edge_nodes[i_sd]
            for j in range(len(E_nodes)):
                E_node = E_nodes[j]
                glob_node = self.sd_nodes[i_sd][E_node]
                sub_vec = E_rhs[3*j:(3*j+3)]
                print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

            print(f"E prev-soln in pc{i_sd=}")
            E_nodes = self.sd_edge_nodes[i_sd]
            for j in range(len(E_nodes)):
                E_node = E_nodes[j]
                glob_node = self.sd_nodes[i_sd][E_node]
                sub_vec = uE0[i_sd][3*j:(3*j+3)]
                print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

            print(f"E soln in pc{i_sd=}")
            E_nodes = self.sd_edge_nodes[i_sd]
            for j in range(len(E_nodes)):
                E_node = E_nodes[j]
                glob_node = self.sd_nodes[i_sd][E_node]
                sub_vec = uE_dict[i_sd][3*j:(3*j+3)]
                print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

T *h_f_V2 = this->f_V.createHostVec().getPtr();
        printf("h_Fv_rhs 2:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_f_V2[lam_dof]);
    }
    printf("\n");
        }

        T *h_f_u0 = this->u_V.createHostVec().getPtr();
        printf("h_uv_soln 2:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_f_u0[lam_dof]);
    }
    printf("\n");
        }

T *h_FE_rhs = this->f_IE.createHostVec().getPtr();
        printf("h_IE_rhs in pc:\n");
        for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    int iglob = this->IE_nodes[ilam];
    if (this->node_class_ind[iglob] != EDGE) continue;
    printf("i_IE %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_FE_rhs[lam_dof]);
    }
    printf("\n");
        }

T *h_IE_rhs = this->u_IE.createHostVec().getPtr();
        printf("h_IE_corr in pc:\n");
        for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    int iglob = this->IE_nodes[ilam];
    if (this->node_class_ind[iglob] != EDGE) continue;
    printf("i_IE %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_IE_rhs[lam_dof]);
    }
    printf("\n");
        }

// debug IE part
        this->addVecIEtoIEV(this->u_IEV, this->u_IE, 1.0, 0.0);
        T *h_IE_soln2 = this->u_IE.createHostVec().getPtr();
        printf("h_IE_soln in pc:\n");
        for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    int iglob = this->IE_nodes[ilam];
    if (this->node_class_ind[iglob] != EDGE) continue;
    printf("i_IE %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_IE_soln2[lam_dof]);
    }
    printf("\n");
        }

T *h_tempV = this->u_V.createHostVec().getPtr();
        printf("h_final_uV:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_tempV[lam_dof]);
    }
    printf("\n");
        }

T *h_gamrhsV = gam_rhs.createHostVec().getPtr();
        printf("h_gamrhsV:\n");
        for (int ilam = 0; ilam < this->Vc_nnodes; ilam++) {
    int iglob = this->Vc_nodes[ilam];
    printf("ivc %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_gamrhsV[n_edge * this->block_dim + lam_dof]);
    }
    printf("\n");
        }

T *h_int_rhs = this->u_I.createHostVec().getPtr();
        printf("h_edge_rhs:\n");
        for (int ilam = 0; ilam < this->I_nnodes; ilam++) {
    int iglob = this->I_nodes[ilam];
    printf("i_int %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_int_rhs[lam_dof]);
    }
    printf("\n");
        }

T *h_gam = gam_out.createHostVec().getPtr();
        printf("h_gam mat-vec:\n");
        for (int ilam = 0; ilam < this->ngam; ilam++) {
    int iglob = this->gam_nodes[ilam];
    printf("\tigam %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_gam[lam_dof]);
    }
    printf("\n");
        }

T *h_gam = gam.createHostVec().getPtr();
        printf("h_gam pc-solve:\n");
        for (int ilam = 0; ilam < this->ngam; ilam++) {
    int iglob = this->gam_nodes[ilam];
    printf("\tigam %d, glob node %d: ", ilam, iglob);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_gam[lam_dof]);
    }
    printf("\n");
        }

print("precond-solve-uE")
        E_nodes = self.edge_interface_nodes_global
        for j in range(len(E_nodes)):
            E_node = E_nodes[j]
            sub_vec = zE[3*j:(3*j+3)]
            print(f"\tedge soln {j=} {E_node=} {sub_vec}")

        V_nodes = self.vertex_nodes_global
        print("precond-solve-uV")
        for j in range(len(V_nodes)):
            v_node = self.vertex_nodes_global[j]
            sub_vec = zV[3*j:(3*j+3)]
            print(f"\tuV vertex {j=} {v_node=} {sub_vec}")

print(f"glob-soln-uI {i_sd=}")
            I_nodes = self.sd_interior_nodes[i_sd]
            for j in range(len(I_nodes)):
                I_node = I_nodes[j]
                glob_node = self.sd_nodes[i_sd][I_node]
                sub_vec = uI[3*j:(3*j+3)]
                print(f"\tint-soln {j=} {glob_node=} {sub_vec}")

            print(f"glob-soln-uE {i_sd=}")
            E_nodes = self.sd_edge_nodes[i_sd]
            for j in range(len(E_nodes)):
                E_node = E_nodes[j]
                glob_node = self.sd_nodes[i_sd][E_node]
                sub_vec = uEi[3*j:(3*j+3)]
                print(f"\tedge soln {j=} {glob_node=} {sub_vec}")

            print(f"glob-soln-uV {i_sd=}")
            V_nodes = self.sd_vertex_nodes[i_sd]
            for j in range(len(V_nodes)):
                V_node = V_nodes[j]
                glob_node = self.sd_nodes[i_sd][V_node]
                sub_vec = uVi[3*j:(3*j+3)]
                print(f"\tuV {j=} {glob_node=} {sub_vec}")

T *h_soln = soln.createHostVec().getPtr();
        printf("h_soln:\n");
        for (int ilam = 0; ilam < this->num_nodes; ilam++) {
    printf("\tglob node %d: ", ilam);
    for (int idof = 2; idof < 5; idof++) {
        int lam_dof = this->block_dim * ilam + idof;
        printf("%.6e,", h_soln[lam_dof]);
    }
    printf("\n");
        }