// dependencies to make element, constitutive objects
#include "TACSShellElementTransform.h"
#include "TACSMaterialProperties.h"
#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"

// this example is based off of examples/crm/crm.cpp in TACS

int main() {
    // make the MPI communicator
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    using T = TacsScalar;
    
    TACSShellTransform *transform = new TACSShellNaturalTransform();

    // set material properties for aluminum (no thermal props input this time)
    TacsScalar rho = 2718;
    TacsScalar specific_heat = 0.0;
    TacsScalar E = 70.0e9;
    TacsScalar nu = 0.3;
    TacsScalar ys = 1e11;
    TacsScalar cte = 10e-6; // double check this value
    TacsScalar kappa = 0.0;
    TACSMaterialProperties *mat = 
          new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);
  
    // create the one element data
    int num_elements = 1;
    int num_geo_nodes = 4;
    int num_vars_nodes = 4;

    // set the xpts randomly for this example
    int32_t num_xpts = 12;
    T *xpts = new T[num_xpts];
    for (int ixpt = 0; ixpt < num_xpts; ixpt++) {
      xpts[ixpt] = 1.0345452 + 2.23123432 * ixpt + 0.323 * ixpt * ixpt;
    }

    // set vars randomly for this example
    int32_t num_vars = 24;
    T *vars = new T[num_vars];
    memset(vars, 0.0, num_vars * sizeof(T));
    T *dvars = new T[num_vars];
    memset(dvars, 0.0, num_vars * sizeof(T));
    T *ddvars = new T[num_vars];
    memset(ddvars, 0.0, num_vars * sizeof(T));

    T *res = new T[num_vars];
    memset(res, 0.0, num_vars * sizeof(T));
    T *Kmat = new T[num_vars*num_vars];
    memset(Kmat, 0.0, num_vars*num_vars*sizeof(T));

    bool nz_vars = true;
    if (nz_vars) {
      for (int ivar = 0; ivar < num_vars; ivar++) {
        vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
      }
    }
    
    TacsScalar thick = 0.005; // shell thickness
    TACSIsoShellConstitutive *con = new TACSIsoShellConstitutive(mat, thick);

    // now create the shell element object
    TACSQuad4Shell *elem = new TACSQuad4Shell(transform, con);

    // call compute energies on the one element
    int elemIndex = 0;
    double time = 0.0;
    // printf("TACS Ue 1 = %.8e\n", *Ue);
    elem->addJacobian(elemIndex, time, 1.0, 0.0, 0.0, xpts, vars, dvars, ddvars, res, Kmat);
    // printf("Analytic jac\n");
    // printf("jac TD = ?\n");

    for (int i = 0; i < 24*24; i++) {
      printf("K[%d] = %.8e\n", i, Kmat[i]);
    }

    MPI_Finalize();

    return 0;
}