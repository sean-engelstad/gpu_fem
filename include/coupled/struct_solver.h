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
    using Mat = Assembler::Mat;
    using Vec = Assembler::VecType<T>;
    BaseTacsInterface(Assembler &assembler, Mat &kmat, LinearSolveFunc linear_solve) : 
        assembler(assembler), kmat(kmat), linear_solve(linear_solve) {
        // create vectors
        loads = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();
    }

    int get_num_vars() {return assembler.get_num_vars();}
    Vec getStructDisps() {return soln;}

    virtual void solve(Vec &struct_loads);

private:
    Assembler assembler;
    LinearSolveFunc linear_solve;
    Mat kmat;
    Vec loads, soln, res, vars, rhs;
}

template <typename T, class Assembler>
class TacsLinearStatic : public BaseTacsStatic<T, Assembler> {
public:
    TacsLinearInterface(Assembler &assembler, Mat &kmat, LinearSolveFunc linear_solve) : BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve) {
        // compute the linear kmat on construction
        assembler.set_variables(vars);
        assembler.add_jacobian(res, kmat);
        assembler.apply_bcs(kmat);
    }

    void solve(Vec& struct_loads) override {
        // copy loads
        loads.zeroValues();
        struct_loads.copyValuesTo(loads);
        assembler.apply_bcs(loads);

        // solve the linear system
        soln.zeroValues();
        linear_solve(kmat, loads, soln);
    }
};

// TODO : later make separate classes or settings to use Newton solver vs. Riks solver here?

template <typename T, class Assembler>
class TacsNonlinearStaticNewton  : public BaseTacsStatic<T, Assembler> {
public:
    using Mat = Assembler::Mat;
    using Vec = Assembler::VecType<T>;

    TacsNonlinearInterface(Assembler &assembler, Mat &kmat, LinearSolveFunc linear_solve,
        int num_load_factors, int num_newton, T abs_tol, T rel_tol, 
        std::string outputFilePrefix = "tacs_output", bool print = false) : 
        BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve) {
        
        // store the nonlinear newton solve settings
        this->num_load_factors = num_load_factors;
        this->num_newton = num_newton;
        this->abs_tol = abs_tol;
        this->rel_tol = rel_tol;
        this->outputFilePrefix = outputFilePrefix;
        this->print = print;
    }

    void solve(Vec& struct_loads) override {
        // copy loads
        loads.zeroValues();
        struct_loads.copyValuesTo(loads);
        assembler.apply_bcs(loads);

        // TODO : add continuation strategy where min_load_factor is not zero each time, and looser solves at start
        // for now just full solve
        T min_load_factor = 0.0;
        T max_load_factor = 1.0;

        // now do Newton solve
        soln.zeroValues();
        newton_solve(linear_solve, kmat, loads, soln, assembler, res, rhs, vars, num_load_factors,
            min_load_factor, max_load_factor, num_newton, abs_tol, rel_tol, outputFilePrefix, print);
    }

private:
    int num_load_factors, num_newton;
    T abs_tol, rel_tol;
    std::string outputFilePrefix;
    bool print;
};
