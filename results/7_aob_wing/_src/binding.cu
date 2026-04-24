// binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector
// #include "lin_cfi_optim.h"         // your solver class header
#include "lin_gmg_cp.h"
#include "lin_gmg_asw.h"
#include "lin_bddc_lu.h"
#include "lin_stiff_optim.h"
// #include "nl_optim.h"        

namespace py = pybind11;

PYBIND11_MODULE(wingmultigrid, m) {
    py::class_<LinearMITC_GMGCP_WingSolver>(m, "LinearMITC_GMGCP_WingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool, int>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false,
             py::arg("n_krylov") = 50)
        .def("set_design_variables", &LinearMITC_GMGCP_WingSolver::set_design_variables)
        .def("get_num_vars",         &LinearMITC_GMGCP_WingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearMITC_GMGCP_WingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearMITC_GMGCP_WingSolver::get_num_lin_solves)
        .def("solve",                &LinearMITC_GMGCP_WingSolver::solve)
        .def("writeSolution",         &LinearMITC_GMGCP_WingSolver::writeSolution)
        .def("writeLoadsToVTK",         &LinearMITC_GMGCP_WingSolver::writeLoadsToVTK)
        .def("writeExplodedVTKs",         &LinearMITC_GMGCP_WingSolver::writeExplodedVTKs)
        .def("evalFunction",         &LinearMITC_GMGCP_WingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearMITC_GMGCP_WingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearMITC_GMGCP_WingSolver::free);

    py::class_<LinearMITC_GMGASW_WingSolver>(m, "LinearMITC_GMGASW_WingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool, int>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false,
             py::arg("n_krylov") = 50)
        .def("set_design_variables", &LinearMITC_GMGASW_WingSolver::set_design_variables)
        .def("get_num_vars",         &LinearMITC_GMGASW_WingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearMITC_GMGASW_WingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearMITC_GMGASW_WingSolver::get_num_lin_solves)
        .def("solve",                &LinearMITC_GMGASW_WingSolver::solve)
        .def("writeSolution",         &LinearMITC_GMGASW_WingSolver::writeSolution)
        .def("writeLoadsToVTK",         &LinearMITC_GMGASW_WingSolver::writeLoadsToVTK)
        .def("writeExplodedVTKs",         &LinearMITC_GMGASW_WingSolver::writeExplodedVTKs)
        .def("evalFunction",         &LinearMITC_GMGASW_WingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearMITC_GMGASW_WingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearMITC_GMGASW_WingSolver::free);

    py::class_<LinearMITC_BDDCLU_WingSolver>(m, "LinearMITC_BDDCLU_WingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool, int>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false,
             py::arg("n_krylov") = 50)
        .def("set_design_variables", &LinearMITC_BDDCLU_WingSolver::set_design_variables)
        .def("get_num_vars",         &LinearMITC_BDDCLU_WingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearMITC_BDDCLU_WingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearMITC_BDDCLU_WingSolver::get_num_lin_solves)
        .def("solve",                &LinearMITC_BDDCLU_WingSolver::solve)
        .def("writeSolution",         &LinearMITC_BDDCLU_WingSolver::writeSolution)
        .def("writeLoadsToVTK",         &LinearMITC_BDDCLU_WingSolver::writeLoadsToVTK)
        .def("writeExplodedVTKs",         &LinearMITC_BDDCLU_WingSolver::writeExplodedVTKs)
        .def("evalFunction",         &LinearMITC_BDDCLU_WingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearMITC_BDDCLU_WingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearMITC_BDDCLU_WingSolver::free);

    py::class_<LinearStiffenedWingSolver>(m, "LinearStiffenedWingSolver")
        // only expose (rhoKS, load_mag)
        .def(py::init<double, double, double, double, int, double, int, int, int, bool>(),
             py::arg("rhoKS")    = 100.0,
             py::arg("safety_factor") = 1.5,
             py::arg("force") = 684e3,
             py::arg("omega") = 0.8,
             py::arg("level") = 2,
             py::arg("rtol") = 1e-6,
             py::arg("ORDER") = 8,
             py::arg("nsmooth") = 1,
             py::arg("ninnercyc") = 1,
             py::arg("print") = false)
        .def("set_design_variables", &LinearStiffenedWingSolver::set_design_variables)
        .def("get_num_vars",         &LinearStiffenedWingSolver::get_num_vars)
        .def("get_num_dvs",          &LinearStiffenedWingSolver::get_num_dvs)
        .def("get_num_lin_solves",          &LinearStiffenedWingSolver::get_num_lin_solves)
        .def("solve",                &LinearStiffenedWingSolver::solve)
        .def("writeSolution",         &LinearStiffenedWingSolver::writeSolution)
        .def("evalFunction",         &LinearStiffenedWingSolver::evalFunction)
        .def("evalFunctionSens",
             [](LinearStiffenedWingSolver &s, const std::string &name) {
                 std::vector<double> sens(s.get_num_dvs());
                 s.evalFunctionSens(name, sens.data());
                 return sens;
             })
        .def("free", &LinearStiffenedWingSolver::free);

    // py::class_<NonlinearPlateSolver>(m, "NonlinearPlateSolver")
    //     // only expose (rhoKS, load_mag)
    //     .def(py::init<double, double, double, double, int, int, int, double, double, double, int, int, double, bool>(),
    //          py::arg("rhoKS")    = 100.0,
    //          py::arg("safety_factor") = 1.5,
    //          py::arg("load_mag") = 100.0,
    //          py::arg("omega") = 1.0,
    //          py::arg("nxe") = 100,
    //          py::arg("nx_comp") = 5,
    //          py::arg("ny_comp") = 5,
    //          py::arg("SR") = 50.0,
    //          py::arg("ORDER") = 8,
    //          py::arg("Lx") = 1.0,
    //          py::arg("nsmooth") = 1,
    //          py::arg("ninnercyc") = 1,
    //          py::arg("in_plane_frac") = 0.1,
    //          py::arg("print") = false)
    //     .def("set_design_variables", &NonlinearPlateSolver::set_design_variables)
    //     .def("get_num_vars",         &NonlinearPlateSolver::get_num_vars)
    //     .def("get_num_dvs",          &NonlinearPlateSolver::get_num_dvs)
    //     .def("get_num_lin_solves",          &NonlinearPlateSolver::get_num_lin_solves)
    //     .def("solve",                &NonlinearPlateSolver::solve)
    //     .def("writeSolution",         &NonlinearPlateSolver::writeSolution)
    //     .def("evalFunction",         &NonlinearPlateSolver::evalFunction)
    //     .def("evalFunctionSens",
    //          [](NonlinearPlateSolver &s, const std::string &name) {
    //              std::vector<double> sens(s.get_num_dvs());
    //              s.evalFunctionSens(name, sens.data());
    //              return sens;
    //          })
    //     .def("free", &NonlinearPlateSolver::free);
}