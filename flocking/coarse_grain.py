"""
Coarse-graining of particle trajectories (dissertation section 2.4).

Reads trajectories.npy (NumPy .npy array, columns [step, index, x, y, vx, vy], rows
ordered by snapshot then particle index) and builds, on a regular grid and for every
saved time, the coarse-grained fields

    rho[y](x, t) = sum_i W_h(|x - x_i(t)|) y_i(t)          (Eq. 2.34)

for y = 1, v_a, v_a v_b, i.e. the fields

    rho            (scalar density,         Eq. 2.35)
    rho[v_a]       (momentum density,       2 components)
    rho[v_a v_b]   (second moment tensor,   2 x 2, symmetric)

Kernel (Eq. 2.33):  W_h(r) = c_h (a_h^2 - r^2)^p  for r < a_h, else 0, with p >= 3
(so that two derivatives are continuous). In n = 2 dimensions the constants that
satisfy  int W dr = 1  (Eq. 2.31)  and  int |r|^2 W dr = h^2  (Eq. 2.32)  are

    a_h = h * sqrt(p + 2),     c_h = (p + 1) / (pi * a_h^(2p + 2)).

Because W has compact support, only particles within a_h of a grid point contribute;
these pairs are found with a k-d tree. As suggested in the text, the fields are by
default rescaled by 1 / (mean number density) so that rho is O(1) (use --no-rescale to
get raw number densities).

Output
------
  fields.npz   (np.load / np.savez_compressed archive)
      x, y            grid-cell centres (1D arrays)
      t               times of the snapshots (step * dt)
      rho             shape (nt, nx, ny)
      rho_v           shape (nt, nx, ny, 2)       rho[v_a],      a = x, y
      rho_vv          shape (nt, nx, ny, 2, 2)    rho[v_a v_b],  symmetric in (a, b)
      h, a_h, p, L, N, rescaled   kernel scale, kernel radius, exponent, box side,
                                  particle number, whether fields were divided by the
                                  mean number density

Use visualize_fields.py to view the stored fields.

Usage:  python coarse_grain.py trajectories.npy --L 126 [--fields-out fields.npz]
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


# ------------------------- kernel -------------------------
def kernel_constants(h, p=3):
    """a_h and c_h for W_h(r) = c_h (a_h^2 - r^2)^p in 2D, unit norm and second moment h^2."""
    a_h = h * np.sqrt(p + 2.0)
    c_h = (p + 1.0) / (np.pi * a_h ** (2 * p + 2))
    return a_h, c_h


# ------------------------- coarse graining of one snapshot -------------------------
def coarse_grain_snapshot(pos, vel, gridpts, h, p, L, periodic_x):
    """Return rho[1], rho[v_x], rho[v_y], rho[v_x v_x], rho[v_x v_y], rho[v_y v_y]
    at every grid point, shape (n_grid, 6)."""
    a_h, c_h = kernel_constants(h, p)
    if periodic_x:   # periodic in x only: large, non-interfering period in y
        box = [L, 4 * L]
        ptree = cKDTree(np.minimum(pos, [np.nextafter(L, 0), 4 * L]), boxsize=box)
        gtree = cKDTree(np.minimum(gridpts, [np.nextafter(L, 0), 4 * L]), boxsize=box)
    else:
        ptree, gtree = cKDTree(pos), cKDTree(gridpts)

    pairs = gtree.sparse_distance_matrix(ptree, a_h, output_type="ndarray")  # fields i, j, v
    W = c_h * (a_h ** 2 - pairs["v"] ** 2) ** p                                # Eq. 2.33
    Wmat = coo_matrix((W, (pairs["i"], pairs["j"])), shape=(len(gridpts), len(pos))).tocsr()

    vx, vy = vel[:, 0], vel[:, 1]
    Y = np.column_stack([np.ones(len(pos)), vx, vy, vx * vx, vx * vy, vy * vy])  # y_i
    return Wmat @ Y                                                             # Eq. 2.34


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", nargs="?", default="trajectories.npy")
    ap.add_argument("--L", type=float, default=None,
                    help="box side (default: int(2*sqrt(N)/0.5), the convention of particles.py)")
    ap.add_argument("--dt", type=float, default=0.01, help="time step used in the simulation")
    ap.add_argument("--periodic-x", action="store_true", help="data are periodic in x")
    ap.add_argument("--h-factor", type=float, default=2.0,
                    help="kernel scale h in units of the mean inter-particle spacing (default 2)")
    ap.add_argument("--p", type=int, default=3, help="kernel exponent, p >= 3 (default 3)")
    ap.add_argument("--grid", type=int, default=100, help="grid points per side")
    ap.add_argument("--no-rescale", action="store_true", help="do not divide by the mean number density")
    ap.add_argument("--fields-out", default="fields.npz")
    args = ap.parse_args()
    if args.p < 3:
        ap.error("p must be >= 3")

    # ---- load data ----
    data = np.load(args.infile)
    if data.ndim != 2 or data.shape[1] < 6:
        raise SystemExit("Expected an array with columns [step, index, x, y, vx, vy]; "
                         "re-run particles.py with velocity output.")
    N = int(data[:, 1].max()) + 1
    traj = data.reshape(-1, N, 6)
    steps = traj[:, 0, 0]
    assert np.all(traj[:, :, 0] == steps[:, None]) and np.all(traj[:, :, 1] == np.arange(N)), \
        "rows must be ordered by snapshot, then particle index"
    t = steps * args.dt
    nt = len(t)

    L = args.L if args.L is not None else float(int(2 * np.sqrt(N) / 0.5))
    n0 = N / L ** 2                              # mean number density
    spacing = 1.0 / np.sqrt(n0)                  # mean inter-particle spacing
    h = args.h_factor * spacing
    a_h, _ = kernel_constants(h, args.p)
    print(f"N = {N}, snapshots = {nt}, L = {L:g}, mean spacing = {spacing:.3f}, "
          f"h = {h:.3f}, kernel radius a_h = {a_h:.3f}")

    # ---- grid (cell centres) ----
    ng = args.grid
    xs = (np.arange(ng) + 0.5) * L / ng
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    gridpts = np.column_stack([X.ravel(), Y.ravel()])

    # ---- coarse-grain every snapshot ----
    scale = 1.0 if args.no_rescale else 1.0 / n0
    rho = np.empty((nt, ng, ng), np.float32)
    rho_v = np.empty((nt, ng, ng, 2), np.float32)
    rho_vv = np.empty((nt, ng, ng, 2, 2), np.float32)
    for k in range(nt):
        f = scale * coarse_grain_snapshot(traj[k, :, 2:4], traj[k, :, 4:6], gridpts,
                                          h, args.p, L, args.periodic_x)
        rho[k] = f[:, 0].reshape(ng, ng)
        rho_v[k] = f[:, 1:3].reshape(ng, ng, 2)
        rho_vv[k, ..., 0, 0] = f[:, 3].reshape(ng, ng)
        rho_vv[k, ..., 0, 1] = rho_vv[k, ..., 1, 0] = f[:, 4].reshape(ng, ng)
        rho_vv[k, ..., 1, 1] = f[:, 5].reshape(ng, ng)
    np.savez_compressed(args.fields_out, x=xs, y=xs, t=t, rho=rho, rho_v=rho_v, rho_vv=rho_vv,
                        h=h, a_h=a_h, p=args.p, L=L, N=N, rescaled=not args.no_rescale)
    print(f"Saved {args.fields_out}: rho {rho.shape}, rho_v {rho_v.shape}, rho_vv {rho_vv.shape}")


if __name__ == "__main__":
    main()
