#include <fstream>

#include "TACSIsoShellConstitutive.h"
#include "TACSKSFailure.h"
#include "TACSMeshLoader.h"
#include "TACSShellElementDefs.h"

/* goal of this script is to writeout CPU precond (pre-factorization) to binary so we can load and
 * factor on GPU */

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    // Write name of BDF file to be load to char array
    const char *filename = "../../../examples/uCRM/CRM_box_2nd.bdf";
    // const char *filename = "../../../examples/performance/uCRM-135_wingbox_fine.bdf";

    // Create the mesh loader object and load file
    TACSMeshLoader *mesh = new TACSMeshLoader(comm);
    mesh->incref();
    mesh->scanBDFFile(filename);

    // Get number of components prescribed in BDF file
    int num_components = mesh->getNumComponents();

    // Set properties needed to create stiffness object
    TacsScalar rho = 2500.0;  // density, kg/m^3
    TacsScalar specific_heat = 921.096;
    TacsScalar E = 70e9;       // elastic modulus, Pa
    TacsScalar nu = 0.3;       // poisson's ratio
    TacsScalar ys = 350e6;     // yield stress, Pa
    TacsScalar cte = 24.0e-6;  // Coefficient of thermal expansion
    TacsScalar kappa = 230.0;  // Thermal conductivity

    TACSMaterialProperties *props =
        new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

    TACSShellTransform *transform = new TACSShellNaturalTransform();

    // Loop over components, creating constituitive object for each
    for (int i = 0; i < num_components; i++) {
        const char *descriptor = mesh->getElementDescript(i);
        TacsScalar thickness = 0.02;
        int thickness_index = i;
        TacsScalar min_thickness = 0.01;
        TacsScalar max_thickness = 0.20;
        TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, thickness, thickness_index,
                                                                  min_thickness, max_thickness);

        // Initialize element object
        TACSElement *shell = TacsCreateShellByName(descriptor, transform, con);

        // Set the shell element
        mesh->setElement(i, shell);
    }

    // Create tacs assembler from mesh loader object
    TACSAssembler *assembler = mesh->createTACS(6);
    assembler->incref();
    mesh->decref();

    // Create matrix and vectors
    TACSBVec *res = assembler->createVec();  // The residual
    // TACS_AMD_ORDER);
    TACSSchurMat *mat = assembler->createSchurMat();  // stiffness matrix

    // Increment reference count to the matrix/vectors
    res->incref();
    mat->incref();

    // Assemble and factor the stiffness/Jacobian matrix. Factor the
    // Jacobian and solve the linear system for the displacements
    bool here1 = false, here2 = true;
    double t2 = MPI_Wtime();
    double alpha = 1.0, beta = 0.0, gamma = 0.0;
    assembler->assembleJacobian(alpha, beta, gamma, NULL, mat);
    double t3 = MPI_Wtime();
    double dt2 = t3 - t2;
    printf("Assembly = %.4e sec\n", dt2);

    // Allocate the factorization
    double t0 = MPI_Wtime();
    int lev = 7;
    double fill = 10.0;
    int reorder_schur = 1;
    TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
    pc->incref();
    // pc->factor(); // will factor on GPU
    double t1 = MPI_Wtime();
    double dt1 = t1 - t0;

    BCSRMat *Bpc, *C, *Emat, *F, *B;
    mat->getBCSRMat(&B, &C, &Emat, &F);
    pc->getBCSRMat(&Bpc, &C, &Emat, &F);

    BCSRMatData *Bdata = B->getMatData();
    int *rowp = Bdata->rowp;
    int *cols = Bdata->cols;
    double *vals = Bdata->A;
    int nrows = Bdata->nrows;
    int ct = 0;

    // copy values from B into Bpc (no factor)
    Bpc->copyValues(B);

    // writeout BCSR precond matrix (before factor) to binary so we can load on GPU and factor there
    std::ofstream fout("cpu_prefactor_ILU7.bcsr", std::ios::binary);

    // at one point was zero
    // printf("vals:");
    // for (int i = 0; i < 800; i++) {
    //     printf("%.2e,", vals[i]);
    // }
    // printf("\n");

    int block_dim = 6;
    int n_block_rows = nrows;
    int nnzb = rowp[nrows];
    fout.write((char *)&block_dim, sizeof(int));
    fout.write((char *)&n_block_rows, sizeof(int));
    fout.write((char *)&nnzb, sizeof(int));

    // Write arrays
    fout.write((char *)rowp, sizeof(int) * (n_block_rows + 1));
    fout.write((char *)cols, sizeof(int) * nnzb);
    fout.write((char *)vals, sizeof(double) * nnzb * block_dim * block_dim);

    fout.close();

    // Decref TACS
    assembler->decref();
    mat->decref();
    // pc->decref();

    MPI_Finalize();

    return 0;
}
