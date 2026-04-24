// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
#include "lin_gmg_cp.h"         // your solver class header
#include "lin_gmg_asw.h"         // your solver class header
#include "lin_bddc_lu.h"         // your solver class header
#include "lin_lu.h"         // your solver class header
#include "nl_gmg_cp.h"        
#include "nl_gmg_asw.h"        
#include "nl_bddc_lu.h"

namespace py = pybind11;

PYBIND11_MODULE(platemultigrid, m) {
    py::class_<Linear_GMGCP_PlateSolver>(m, "Linear_GMGCP_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Linear_GMGCP_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Linear_GMGCP_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Linear_GMGCP_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Linear_GMGCP_PlateSolver::get_num_lin_solves)
        .def("solve",                &Linear_GMGCP_PlateSolver::solve)
        .def("writeSolution",         &Linear_GMGCP_PlateSolver::writeSolution)
        .def("evalFunction",         &Linear_GMGCP_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Linear_GMGCP_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Linear_GMGCP_PlateSolver::free);

    py::class_<Linear_GMGASW_PlateSolver>(m, "Linear_GMGASW_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Linear_GMGASW_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Linear_GMGASW_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Linear_GMGASW_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Linear_GMGASW_PlateSolver::get_num_lin_solves)
        .def("solve",                &Linear_GMGASW_PlateSolver::solve)
        .def("writeSolution",         &Linear_GMGASW_PlateSolver::writeSolution)
        .def("evalFunction",         &Linear_GMGASW_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Linear_GMGASW_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Linear_GMGASW_PlateSolver::free);

    py::class_<Linear_BDDCLU_PlateSolver>(m, "Linear_BDDCLU_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Linear_BDDCLU_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Linear_BDDCLU_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Linear_BDDCLU_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Linear_BDDCLU_PlateSolver::get_num_lin_solves)
        .def("solve",                &Linear_BDDCLU_PlateSolver::solve)
        .def("writeSolution",         &Linear_BDDCLU_PlateSolver::writeSolution)
        .def("evalFunction",         &Linear_BDDCLU_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Linear_BDDCLU_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Linear_BDDCLU_PlateSolver::free);

    py::class_<Linear_LU_PlateSolver>(m, "Linear_LU_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, int, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Linear_LU_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Linear_LU_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Linear_LU_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Linear_LU_PlateSolver::get_num_lin_solves)
        .def("solve",                &Linear_LU_PlateSolver::solve)
        .def("writeSolution",         &Linear_LU_PlateSolver::writeSolution)
        .def("evalFunction",         &Linear_LU_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Linear_LU_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Linear_LU_PlateSolver::free);

    py::class_<Nonlinear_GMGCP_PlateSolver>(m, "Nonlinear_GMGCP_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Nonlinear_GMGCP_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Nonlinear_GMGCP_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Nonlinear_GMGCP_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Nonlinear_GMGCP_PlateSolver::get_num_lin_solves)
        .def("solve",                &Nonlinear_GMGCP_PlateSolver::solve)
        .def("writeSolution",         &Nonlinear_GMGCP_PlateSolver::writeSolution)
        .def("evalFunction",         &Nonlinear_GMGCP_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Nonlinear_GMGCP_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Nonlinear_GMGCP_PlateSolver::free);

    py::class_<Nonlinear_GMGASW_PlateSolver>(m, "Nonlinear_GMGASW_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Nonlinear_GMGASW_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Nonlinear_GMGASW_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Nonlinear_GMGASW_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Nonlinear_GMGASW_PlateSolver::get_num_lin_solves)
        .def("solve",                &Nonlinear_GMGASW_PlateSolver::solve)
        .def("writeSolution",         &Nonlinear_GMGASW_PlateSolver::writeSolution)
        .def("evalFunction",         &Nonlinear_GMGASW_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Nonlinear_GMGASW_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Nonlinear_GMGASW_PlateSolver::free);

    py::class_<Nonlinear_BDDCLU_PlateSolver>(m, "Nonlinear_BDDCLU_PlateSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("load_mag") = 100.0,
             py::arg("omega") = 1.0,
             py::arg("nxe") = 100,
             py::arg("nx_comp") = 5,
             py::arg("ny_comp") = 5,
             py::arg("SR") = 50.0,
             py::arg("ORDER") = 8,
             py::arg("Lx") = 1.0,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("in_plane_frac") = 0.1,
             py::arg("print") = false)
        .def("set_design_variables", &Nonlinear_BDDCLU_PlateSolver::set_design_variables)
        .def("get_num_vars",         &Nonlinear_BDDCLU_PlateSolver::get_num_vars)
        .def("get_num_dvs",          &Nonlinear_BDDCLU_PlateSolver::get_num_dvs)
        .def("get_num_lin_solves",          &Nonlinear_BDDCLU_PlateSolver::get_num_lin_solves)
        .def("solve",                &Nonlinear_BDDCLU_PlateSolver::solve)
        .def("writeSolution",         &Nonlinear_BDDCLU_PlateSolver::writeSolution)
        .def("evalFunction",         &Nonlinear_BDDCLU_PlateSolver::evalFunction)
        .def("evalFunctionSens",
             [](Nonlinear_BDDCLU_PlateSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &Nonlinear_BDDCLU_PlateSolver::free);
}