#pragma once
#include <stdexcept>
#include <string>

#include "linalg/vec.h"

template <typename T, template <typename> class Vec>
class Function {
   public:
    Function(std::string& name) : name(name), setup(false) { *value = 0.0; }
    std::string name;
    T* value;
    bool setup;

    void check_setup() {
        if (!setup) {
            throw std::runtime_error("Function is not set up properly!");
        }
    }

    void init_sens(int num_dvs, int num_xpts) {
        dv_sens = Vec<T>(num_dvs);
        xpt_sens = Vec<T>(num_xpts);
        setup = true;
    }

    //    private:
    // should we make this hostvec later?
    Vec<T> dv_sens;
    Vec<T> xpt_sens;
};

template <typename T, template <typename> class Vec>
class KSFailure : public Function<T> {
   public:
    KSFailure(T rho_KS) : Function("ksfailure"), rho_KS(rho_KS) {}
    T rho_KS;
};

template <typename T, template <typename> class Vec>
class Mass : public Function<T> {
   public:
    Mass() : Function("mass") {}
};