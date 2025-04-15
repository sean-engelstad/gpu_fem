
template <typename T, class Vec>
class BaseAeroSolver {
    // TODO:
};

template <typename T, class Vec>
class FixedAeroSolver {
    // supplies fixed aero loads each time
public:
    FixedAeroSolver(int na_surf, Vec &fa) : na_surf(na_surf), fa(fa) {}

    void solve(Vec &ua) {}
    Vec getAeroLoads() { return fa; }

    int get_num_surface_vars() { return na_surf; }
private:
    int na_surf;
    Vec fa;
};