"""
Constant-speed particles in a square box with Cucker-Smale alignment and close-range repulsion.

Setup
-----
* Box side L. Walls at y = 0 and y = L always reflect specularly. In x the
  box is either periodic (PERIODIC_X = True, minimum-image convention) or has
  reflecting walls at x = 0 and x = L too (PERIODIC_X = False).
  There are no fixed particles.
* N mobile particles move with constant speed v.
* Newton's second law with the force projected onto the orthogonal complement
  of the velocity:

      m dv/dt = F_perp = F - (F . v_hat) v_hat

  (|v| is also re-normalised each step to remove the O(dt^2) drift.)
* Cucker-Smale alignment interaction:

      F_i = sum_j K(r_ij) (v_j - v_i),     K(r) = beta * (1 - r/rc)**2  for r < rc, else 0

  with rc = 3a.
* Close-range repulsion (pair potential, zero at r = 3a, divergent at r = a):

      U(r) = eps * ((3a - r) / (r - a))**2     for a < r < 3a, else 0
      F_rep(r) = -dU/dr = 4*eps*a*(3a - r) / (r - a)**3,  along the line joining
      the particles, pushing them apart.

  Both interactions have strictly finite range rc = 3a, so pairs are found with a
  k-d tree (scipy cKDTree.query_pairs), giving O(N) cost per step at fixed density
  instead of O(N^2). Pair forces are equal and opposite and are applied to both
  partners.

  Total force: F_i = F_align + F_rep. Only the component of F_i perpendicular to v_i survives the
  projection, so the interaction rotates v_i toward its neighbours' velocities.
  The title shows the polar order parameter  Phi = |<v>| / v  (0 = disordered,
  1 = fully aligned flock).

Usage
-----
  python particles.py                   live window (close it to stop)
  python particles.py --T 200           integrate up to time T without plotting and
                                        save trajectories to trajectories.npy
  python particles.py --T 200 --save-every 20 --out data.npy

Output file: a NumPy array in .npy binary format (despite the .dat name), shape
(n_snapshots * N, 6), float64, one row per particle per saved snapshot:

  columns = [step, index, x, y, vx, vy]  (time = step * dt)

Step 0 (initial condition) and the final step are always included. Read it with

  data = np.load("trajectories.npy")
  traj = data.reshape(-1, N, 6)          # traj[snapshot, particle] = [step, index, x, y, vx, vy]
"""
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import EllipseCollection
from scipy.spatial import cKDTree

# ------------------------- parameters -------------------------
a = 1.0            # characteristic length (particle separation)
N = 1000           # number of particles (make 10^4 or more to generate "continuous" results)
density = 0.5;     # particle density (relative to a, should be less than about 1)
L = int(2*N**0.5/density)  # box side
v = 1.0            # speed (constant)
m = 1              # particle mass (decrease to generate density/orientation waves)
beta = 5.0           # alignment strength
eps = 1.0          # repulsion strength
dt = 0.01          # time step
steps_per_frame = 5   # the plot is refreshed every this many time steps
seed = 1
PERIODIC_X = False  # True: periodic in x;  False: reflecting walls on all four sides
rc = 3.0 * a       # cutoff
r_min = 0.02 * a   # regularisation: (r - a) never smaller than this
rng = np.random.default_rng(seed)

# ---- trajectory output (headless mode) ----
T = None                       # integrate until this time and save; None = live plot
save_every = 20               # save positions every this many time steps
out_file = "trajectories.npy"  # output file name


def min_image(d):
    """Minimum-image convention for the x-component (only if periodic in x)."""
    if PERIODIC_X:
        d[..., 0] -= L * np.round(d[..., 0] / L)
    return d


# ------------------------- initial condition -------------------------
def init_positions(n, d_min=1.3 * a):
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError("Cannot place particles: reduce density")
        x0 = rng.uniform(0, L) if PERIODIC_X else rng.uniform(0.5 * a, L - 0.5 * a)
        p = np.array([x0, rng.uniform(0.5 * a, L - 0.5 * a)])
        if pts and np.min(np.linalg.norm(min_image(np.array(pts) - p), axis=1)) < d_min:
            continue
        pts.append(p)
    return np.array(pts)


pos = init_positions(N)
phi0 = rng.uniform(0, 2 * np.pi, N)
vel = v * np.c_[np.cos(phi0), np.sin(phi0)]


# ------------------------- dynamics -------------------------
def forces(pos, vel):
    """Cucker-Smale alignment + close-range repulsion, using a k-d tree for the neighbour search."""
    # Periodic in x only: give the tree a periodic box in x and a non-interfering
    # (much larger) period in y, so no wrap-around occurs in y.
    if PERIODIC_X:
        tree = cKDTree(np.minimum(pos, [np.nextafter(L, 0), 4 * L]), boxsize=[L, 4 * L])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(rc, output_type="ndarray")      # all (i, j), i < j, r < rc
    F = np.zeros_like(pos)
    if len(pairs) == 0:
        return F
    i, j = pairs[:, 0], pairs[:, 1]
    d = min_image(pos[i] - pos[j])                           # r_i - r_j
    r = np.linalg.norm(d, axis=1)
    K = beta * (1.0 - r / rc) ** 2                             # alignment kernel
    x = np.maximum(r - a, r_min)
    rep = 4 * eps * a * (rc - r) / x**3                      # repulsive force magnitude
    f = K[:, None] * (vel[j] - vel[i]) + (rep / r)[:, None] * d   # force on i from j
    for k in (0, 1):                                         # F_i += f, F_j -= f
        F[:, k] = np.bincount(i, f[:, k], len(pos)) - np.bincount(j, f[:, k], len(pos))
    return F


def step(pos, vel):
    F = forces(pos, vel)
    vhat = vel / np.linalg.norm(vel, axis=1)[:, None]
    F_perp = F - np.sum(F * vhat, axis=1)[:, None] * vhat   # project out parallel part
    dv = dt * F_perp / m                                     # m dv/dt = F_perp
    dvn = np.linalg.norm(dv, axis=1)                         # cap for very close encounters
    big = dvn > 0.25 * v
    dv[big] *= (0.25 * v / dvn[big])[:, None]
    vel = vel + dv
    vel = v * vel / np.linalg.norm(vel, axis=1)[:, None]     # keep |v| = v exactly
    pos = pos + dt * vel
    if PERIODIC_X:
        pos[:, 0] %= L                                       # periodic in x
        axes = (1,)
    else:
        axes = (0, 1)                                        # walls on all four sides
    for i in axes:                                           # specular reflection
        lo = pos[:, i] < 0
        hi = pos[:, i] > L
        pos[lo, i] = -pos[lo, i]
        pos[hi, i] = 2 * L - pos[hi, i]
        vel[lo | hi, i] *= -1
    return pos, vel


def heading(vel):
    return np.arctan2(vel[:, 1], vel[:, 0]) % (2 * np.pi)


# ------------------------- integrate to time T and save -------------------------
def run_to_T(T, save_every, out_file):
    """Integrate until time T (no plotting) and save a NumPy array with rows
    [step, index, x, y, vx, vy] for every particle at step 0, every `save_every` steps, and the last step."""
    p, w = pos.copy(), vel.copy()
    nsteps = int(round(T / dt))
    save_steps = sorted(set([0, nsteps] + list(range(save_every, nsteps, save_every))))
    idx = np.arange(N)
    out = np.empty((len(save_steps), N, 6))
    k = 0
    for n in range(nsteps + 1):
        if n > 0:
            p, w = step(p, w)
        if n == save_steps[k]:
            out[k, :, 0] = n
            out[k, :, 1] = idx
            out[k, :, 2:4] = p
            out[k, :, 4:6] = w
            k += 1
    with open(out_file, "wb") as f:           # file handle: keeps the exact name (no .npy appended)
        np.save(f, out.reshape(-1, 6))
    print(f"Wrote {out_file}: {nsteps} steps (t = 0 .. {nsteps * dt:g}), "
          f"{len(save_steps)} snapshots, array shape {(len(save_steps) * N, 6)}")


# ------------------------- live plot -------------------------
def run_live():
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.set_xlim(-a, L + a)
    ax.set_ylim(-a, L + a)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([0, L], [0, 0], color="k", lw=1.5)               # reflecting walls
    ax.plot([0, L], [L, L], color="k", lw=1.5)
    side_style = dict(color="0.6", lw=0.8, ls="--") if PERIODIC_X else dict(color="k", lw=1.5)
    ax.plot([0, 0], [0, L], **side_style)                    # dashed seams or solid walls
    ax.plot([L, L], [0, L], **side_style)
    if PERIODIC_X:
        SHIFT = np.array([[0.0, 0.0], [L, 0.0], [-L, 0.0]])  # periodic images (ghosts)
    else:
        SHIFT = np.array([[0.0, 0.0]])
    NIMG = len(SHIFT)

    def with_images(p):
        return np.concatenate([p + sh for sh in SHIFT])

    cmap = plt.get_cmap("hsv")
    mob = EllipseCollection(a * np.ones(NIMG * N), a * np.ones(NIMG * N), 0, units="xy",
                            offsets=with_images(pos), offset_transform=ax.transData,
                            facecolors=np.tile(cmap(heading(vel) / (2 * np.pi)), (NIMG, 1)),
                            edgecolors="k", linewidths=0.4)
    ax.add_collection(mob)
    quiv = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0] / v, vel[:, 1] / v,
                     angles="xy", scale_units="xy", scale=1.0 / (0.9 * a), width=0.003)
    title = ax.set_title("t = 0.00,  order = 0.00")

    state = {"pos": pos.copy(), "vel": vel.copy(), "t": 0.0}

    def update(frame):
        p, w = state["pos"], state["vel"]
        for _ in range(steps_per_frame):
            p, w = step(p, w)
        state["pos"], state["vel"] = p, w
        state["t"] += steps_per_frame * dt
        mob.set_offsets(with_images(p))
        mob.set_facecolors(np.tile(cmap(heading(w) / (2 * np.pi)), (NIMG, 1)))
        quiv.set_offsets(p)
        quiv.set_UVC(w[:, 0] / v, w[:, 1] / v)
        order = np.linalg.norm(w.mean(axis=0)) / v
        title.set_text(f"t = {state['t']:.2f},  order = {order:.2f}")
        return mob, quiv, title

    anim = animation.FuncAnimation(fig, update, frames=itertools.count(),
                                   interval=20, blit=False, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("--T", type=float, default=T, help="integrate until time T and save trajectories (no plot)")
    parser.add_argument("--save-every", type=int, default=save_every, help="save every this many steps")
    parser.add_argument("--out", default=out_file, help="output file name")
    args = parser.parse_args()
    if args.T is not None:
        run_to_T(args.T, args.save_every, args.out)
    else:
        run_live()
