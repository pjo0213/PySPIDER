"""
Interactive visualisation of the coarse-grained fields produced by coarse_grain.py.

Reads fields.npz (arrays rho, rho_v, rho_vv, t, L, rescaled, ...) and animates the chosen
field/component vs time in a matplotlib window:

  * Play/Pause button or space bar
  * time slider, or left/right arrow keys, to step through the snapshots (pauses playback)

Field choice (--field):
  rho                            scalar density
  rho_vx, rho_vy                 components of rho[v_a]
  rho_vxvx, rho_vxvy, rho_vyvy   components of rho[v_a v_b]

Signed fields use a diverging colour map symmetric about 0; non-negative fields start at 0.
The colour scale is fixed over time (99.5th percentile over all snapshots).

Usage:  python visualize_fields.py [fields.npz] [--field rho_vx] [--interval 60]
"""
import argparse
import numpy as np


# ------------------------- selectable fields -------------------------
# name: (label, signed?, getter(rho, rho_v, rho_vv) -> array of shape (nt, nx, ny))
FIELDS = {
    "rho":      (r"\rho",             False, lambda r, rv, rvv: r),
    "rho_vx":   (r"\rho[v_x]",        True,  lambda r, rv, rvv: rv[..., 0]),
    "rho_vy":   (r"\rho[v_y]",        True,  lambda r, rv, rvv: rv[..., 1]),
    "rho_vxvx": (r"\rho[v_x v_x]",    False, lambda r, rv, rvv: rvv[..., 0, 0]),
    "rho_vxvy": (r"\rho[v_x v_y]",    True,  lambda r, rv, rvv: rvv[..., 0, 1]),
    "rho_vyvy": (r"\rho[v_y v_y]",    False, lambda r, rv, rvv: rvv[..., 1, 1]),
}


def show_field(name, rho, rho_v, rho_vv, t, L, rescaled, interval=60):
    """Animate the chosen field/component vs time in a matplotlib window with
    Play/Pause, a time slider and key bindings."""
    import itertools
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.widgets import Button, Slider

    nt = len(t)
    core, signed, getter = FIELDS[name]
    data = getter(rho, rho_v, rho_vv)                      # (nt, nx, ny)
    if signed:                                             # symmetric diverging scale about 0
        vmax = float(np.percentile(np.abs(data), 99.5)) or 1.0
        vmin, cmap = -vmax, "RdBu_r"
    else:
        vmax, vmin, cmap = float(np.percentile(data, 99.5)), 0.0, "viridis"
    label = rf"${core} / \bar n$" if rescaled else rf"${core}$"

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    fig.subplots_adjust(bottom=0.22)
    im = ax.imshow(data[0].T, origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ttl = ax.set_title(f"t = {t[0]:.2f}")

    state = {"k": 0, "playing": nt > 1}
    slider = None

    def draw(k):
        state["k"] = k
        im.set_data(data[k].T)
        ttl.set_text(f"t = {t[k]:.2f}")
        fig.canvas.draw_idle()

    def set_slider(k):
        slider.eventson = False            # avoid re-triggering on_changed
        slider.set_val(k)
        slider.eventson = True

    def tick(_):
        if state["playing"]:
            k = (state["k"] + 1) % nt
            draw(k)
            if slider is not None:
                set_slider(k)

    if nt > 1:
        sax = fig.add_axes([0.15, 0.10, 0.60, 0.03])
        slider = Slider(sax, "snapshot", 0, nt - 1, valinit=0, valstep=1)

        def on_slide(val):
            state["playing"] = False       # scrubbing pauses the animation
            button.label.set_text("Play")
            draw(int(val))

        slider.on_changed(on_slide)

    bax = fig.add_axes([0.82, 0.07, 0.12, 0.07])
    button = Button(bax, "Pause" if state["playing"] else "Play")

    def toggle(_=None):
        state["playing"] = not state["playing"]
        button.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    button.on_clicked(toggle)

    def on_key(event):
        if event.key == " ":
            toggle()
        elif event.key in ("left", "right") and nt > 1:
            state["playing"] = False
            button.label.set_text("Play")
            k = (state["k"] + (1 if event.key == "right" else -1)) % nt
            draw(k)
            set_slider(k)

    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = animation.FuncAnimation(fig, tick, frames=itertools.count(), interval=interval,
                                   blit=False, cache_frame_data=False)
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", nargs="?", default="fields.npz", help="file written by coarse_grain.py")
    ap.add_argument("--field", choices=list(FIELDS), default="rho",
                    help="field/component to display (default: rho)")
    ap.add_argument("--interval", type=int, default=60, help="animation delay between frames in ms")
    args = ap.parse_args()

    f = np.load(args.infile)
    for key in ("rho", "rho_v", "rho_vv", "t", "L", "rescaled"):
        if key not in f:
            raise SystemExit(f"{args.infile} does not look like coarse_grain.py output (missing '{key}')")
    print(f"{args.infile}: {len(f['t'])} snapshots, grid {f['rho'].shape[1]}x{f['rho'].shape[2]}, "
          f"t = {f['t'][0]:g} .. {f['t'][-1]:g}, showing '{args.field}'")
    show_field(args.field, f["rho"], f["rho_v"], f["rho_vv"], f["t"], float(f["L"]),
               bool(f["rescaled"]), args.interval)


if __name__ == "__main__":
    main()
