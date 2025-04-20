#pragma once
#include <string>

// supported classes
// TacsLinearStatic
// TacsNonlinearStaticNewton
// required methods: solve(fs) => us

template <class Mat, class Vec>
using LinearSolveFunc = void (*)(Mat &, Vec &, Vec &, bool);

template <typename T, class Assembler>
class BaseTacsStatic {
   public:
    using Vec = typename Assembler::template VecType<T>;
    using Mat = typename Assembler::template MatType<Vec>;

    using LinearSolve = LinearSolveFunc<Mat, Vec>;

    BaseTacsStatic(Assembler &assembler, Mat &kmat, LinearSolve linear_solve)
        : assembler(assembler), kmat(kmat), linear_solve(linear_solve) {
        // create vectors
        loads = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();
    }

    int get_num_nodes() { return assembler.get_num_nodes(); }
    Vec getStructDisps() { return vars.removeRotationalDOF(); }
    Assembler &getAssembler() { return assembler; }
    void resetSoln() {
        vars.zeroValues();
        assembler.set_variables(vars);
    }

    // void solve(Vec &struct_loads);  // virtual

   protected:
    Assembler assembler;
    LinearSolve linear_solve;
    Mat kmat;
    Vec loads, soln, res, vars, rhs;
};

template <typename T, class Assembler>
class TacsLinearStatic : public BaseTacsStatic<T, Assembler> {
   public:
    using Base = BaseTacsStatic<T, Assembler>;
    using Vec = typename Base::Vec;
    using Mat = typename Base::Mat;
    using LinearSolve = typename Base::LinearSolve;

    TacsLinearStatic(Assembler &assembler, Mat &kmat, LinearSolve linear_solve, bool print = false)
        : BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve) {
        // compute the linear kmat on construction
        assembler.set_variables(this->vars);
        assembler.add_jacobian(this->res, this->kmat);
        assembler.apply_bcs(this->kmat);
        this->print = print;
    }

    void solve(Vec &struct_loads) {
        // copy loads
        this->loads.zeroValues();
        struct_loads.copyValuesTo(this->loads);
        this->assembler.apply_bcs(this->loads);

        // assemble kmat again for debug
        // TODO : can I preserve cusparse LU factorization? Maybe not..
        this->assembler.set_variables(this->vars);
        this->assembler.add_jacobian(this->res, this->kmat);
        this->assembler.apply_bcs(this->kmat);

        // solve the linear system
        this->soln.zeroValues();
        this->linear_solve(this->kmat, this->loads, this->soln, this->print);
        this->vars.zeroValues();
        this->soln.copyValuesTo(this->vars);
    }

   private:
    bool print;
};

// TODO : later make separate classes or settings to use Newton solver vs. Riks solver here?

template <typename T, class Assembler>
class TacsNonlinearStaticNewton : public BaseTacsStatic<T, Assembler> {
   public:
    using Base = BaseTacsStatic<T, Assembler>;
    using Vec = typename Base::Vec;
    using Mat = typename Base::Mat;
    using LinearSolve = typename Base::LinearSolve;

    TacsNonlinearStaticNewton(Assembler &assembler, Mat &kmat, LinearSolve linear_solve,
                              int num_load_factors, int num_newton, bool print = false,
                              T abs_tol = 1e-8, T rel_tol = 1e-6,
                              std::string outputFilePrefix = "tacs_output")
        : BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve) {
        // store the nonlinear newton solve settings
        this->num_load_factors = num_load_factors;
        this->num_newton = num_newton;
        this->abs_tol = abs_tol;
        this->rel_tol = rel_tol;
        this->outputFilePrefix = outputFilePrefix;
        this->print = print;
    }

    void solve(Vec &struct_loads) {
        // copy loads
        this->loads.zeroValues();
        struct_loads.copyValuesTo(this->loads);
        this->assembler.apply_bcs(this->loads);

        // reset things:
        this->soln.zeroValues();
        this->res.zeroValues();
        this->rhs.zeroValues();

        // TODO : add continuation strategy where min_load_factor is not zero each time, and looser
        // solves at start for now just full solve
        T min_load_factor = 1.0 / num_load_factors;
        T max_load_factor = 1.0;

        // now do Newton solve
        this->soln.zeroValues();
        newton_solve(this->linear_solve, this->kmat, this->loads, this->soln, this->assembler,
                     this->res, this->rhs, this->vars, num_load_factors, min_load_factor,
                     max_load_factor, num_newton, abs_tol, rel_tol, outputFilePrefix, print);
    }

   private:
    int num_load_factors, num_newton;
    T abs_tol, rel_tol;
    std::string outputFilePrefix;
    bool print;
};
