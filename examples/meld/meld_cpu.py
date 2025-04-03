import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import matplotlib.pyplot as plt

np.random.seed(1234567)

def make_grid_mesh(nnx, nny, Lx, Ly, z0):
    N = nnx * nny
    dx = Lx / (nnx - 1)
    dy = Ly / (nny - 1)
    x0 = np.zeros((3 * N,))

    for iy in range(nny):
        yfac = np.sin(np.pi * iy / (nny - 1))
        for ix in range(nnx):
            xfac = np.sin(np.pi * ix / (nnx - 1))
            ind = iy * nnx + ix
            x0[3 * ind] = ix * dx
            x0[3*ind+1] = iy * dy
            x0[3*ind+2] = z0 * xfac * yfac
    
    return x0

def make_inplane_shear_disp(x0, angleDeg):
    N = x0.shape[0] // 3
    u = np.zeros((3 * N,))
    angleRad = angleDeg * np.pi / 180.0

    for inode in range(N):
        u[3*inode] = np.tan(angleRad / 2.0) * x0[3*inode+1]
        u[3*inode+1] = np.tan(angleRad / 2.0) * x0[3*inode]
        u[3*inode+2] = 0.0
    return u

def plot_quantity(xpts, vars, title:str):
    # assume xpts, vars are same size and only plot xy plane behavior
    # with subplots
    fig, ax = plt.subplots(2, 1)
    N = xpts.shape[0] // 3
    n = int(np.sqrt(N))
    X = xpts[0::3].reshape((n,n))
    Y = xpts[1::3].reshape((n,n))
    plt.title(title)
    for i in range(2):
        vd = vars[i::3].reshape((n,n))
        ax[i].contourf(X, Y, vd)
    # plt.show()
    plt.savefig(f"out/{title}.png", dpi=400)


if __name__=="__main__":
    Lx = Ly = 1.0
    nna = 30
    nns = 17
    xs0 = make_grid_mesh(nns, nns, Lx, Ly, 0.01)
    xa0 = make_grid_mesh(nna, nna, Lx, Ly, 0.01)
    na = xa0.shape[0] // 3; ns = xs0.shape[0] // 3

    us = make_inplane_shear_disp(xs0, 20.0)
    fa = make_inplane_shear_disp(xa0, 10.0)
    ua = xa0 * 0.0
    fs = us * 0.0

    # make MELD object
    comm = MPI.COMM_WORLD
    isymm = -1; nn = 32; beta = 10.0
    meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, nn, beta)

    meld.setAeroNodes(xa0)
    meld.setStructNodes(xs0)
    meld.initialize()

    # set displacements
    meld.transferDisps(us, ua)
    meld.transferLoads(fa, fs)

    plot_quantity(xs0, us, r"$u_s$")
    plot_quantity(xa0, ua, r"$u_a$")
    plot_quantity(xa0, fa, r"$f_a$")
    plot_quantity(xs0, fs, r"$f_s$")