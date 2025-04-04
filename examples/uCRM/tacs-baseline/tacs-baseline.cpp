#include "TACSIsoShellConstitutive.h"
#include "TACSKSFailure.h"
#include "TACSMeshLoader.h"
#include "TACSShellElementDefs.h"

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

#ifdef TACS_USE_COMPLEX
    double dh = 1e-30;
#else
    double dh = 1e-6;
#endif // TACS_USE_COMPLEX

    // Scan the arguments to check for a manual step size
    for (int k = 0; k < argc; k++) {
        if (sscanf(argv[k], "dh=%lf", &dh) == 1) {
            if (rank == 0) {
                printf("Using step size dh = %e\n", dh);
            }
        }
    }

    // Write name of BDF file to be load to char array
    // const char *filename = "../CRM_box_2nd.bdf";
    // const char *filename = "../uCRM-135_wingbox_medium.bdf";
    const char *filename = "../uCRM-135_wingbox_fine.bdf";

    // Create the mesh loader object and load file
    TACSMeshLoader *mesh = new TACSMeshLoader(comm);
    mesh->incref();
    mesh->scanBDFFile(filename);

    // Get number of components prescribed in BDF file
    int num_components = mesh->getNumComponents();

    // Set properties needed to create stiffness object
    TacsScalar rho = 2500.0; // density, kg/m^3
    TacsScalar specific_heat = 921.096;
    TacsScalar E = 70e9;      // elastic modulus, Pa
    TacsScalar nu = 0.3;      // poisson's ratio
    TacsScalar ys = 350e6;    // yield stress, Pa
    TacsScalar cte = 24.0e-6; // Coefficient of thermal expansion
    TacsScalar kappa = 230.0; // Thermal conductivity

    TACSMaterialProperties *props =
        new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

    TACSShellTransform *transform = new TACSShellNaturalTransform();

    // Loop over components, creating constituitive object for each
    for (int i = 0; i < num_components; i++) {
        const char *descriptor = mesh->getElementDescript(i);
        TacsScalar thickness = 0.01;
        int thickness_index = i;
        TacsScalar min_thickness = 0.01;
        TacsScalar max_thickness = 0.20;
        TACSShellConstitutive *con = new TACSIsoShellConstitutive(
            props, thickness, thickness_index, min_thickness, max_thickness);

        // Initialize element object
        TACSElement *shell = TacsCreateShellByName(descriptor, transform, con);

        // Set the shell element
        mesh->setElement(i, shell);
    }

    // Create tacs assembler from mesh loader object
    TACSAssembler *assembler = mesh->createTACS(6);
    assembler->incref();
    mesh->decref();

    // Create the design vector
    TACSBVec *x = assembler->createDesignVec();
    x->incref();

    // Get the design variable values
    assembler->getDesignVars(x);

    // Create matrix and vectors
    TACSBVec *ans = assembler->createVec(); // displacements and rotations
    TACSBVec *f = assembler->createVec();   // loads
    TACSBVec *res = assembler->createVec(); // The residual
    TACSSchurMat *mat = assembler->createSchurMat(); // stiffness matrix

    // Increment reference count to the matrix/vectors
    ans->incref();
    f->incref();
    res->incref();
    mat->incref();

    // Allocate the factorization
    double t0 = MPI_Wtime();
    int lev = 10000;
    double fill = 10.0;
    int reorder_schur = 1;
    TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
    pc->incref();
    double t1 = MPI_Wtime();
    double dt1 = t1 - t0;
    printf("Factorization = %.4e sec\n", dt1);

    // Set all the entries in load vector to specified value
    TacsScalar *force_vals;
    int size = f->getArray(&force_vals);
    for (int k = 2; k < size; k += 6) {
        force_vals[k] += 100.0;
    }
    assembler->applyBCs(f);

    // Assemble and factor the stiffness/Jacobian matrix. Factor the
    // Jacobian and solve the linear system for the displacements
    double t2 = MPI_Wtime();
    double alpha = 1.0, beta = 0.0, gamma = 0.0;
    assembler->assembleJacobian(alpha, beta, gamma, NULL, mat);
    double t3 = MPI_Wtime();
    double dt2 = t3 - t2;
    printf("Assembly = %.4e sec\n", dt2);

    pc->factor(); // LU factorization of stiffness matrix
    pc->applyFactor(f, ans);
    assembler->setVariables(ans);

    double t4 = MPI_Wtime();
    double dt3 = t4 - t3;
    printf("Solve time = %.4e\n", dt3);

    // Create an TACSToFH5 object for writing output to files
    int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES |
                      TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                      TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
    TACSToFH5 *f5 =
        new TACSToFH5(assembler, TACS_BEAM_OR_SHELL_ELEMENT, write_flag);
    f5->incref();
    f5->writeToFile("ucrm.f5");
    f5->decref();

    // Free the design variable information
    x->decref();

    // Decref the matrix/pc objects
    mat->decref();
    pc->decref();

    // Decref the vectors
    ans->decref();
    f->decref();
    res->decref();

    // Decref TACS
    assembler->decref();

    MPI_Finalize();

    return 0;
}
