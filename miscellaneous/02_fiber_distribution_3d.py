import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# Utility functions
# ============================================================

def weyl_points(n, alpha=(np.sqrt(2), np.sqrt(3))):
    """
    2D Weyl low-discrepancy sequence in [0,1)^2.
    """
    i = np.arange(1, n + 1)[:, None]
    alpha = np.array(alpha)[None, :]
    return (i * alpha) % 1.0


def sample_points_in_unit_ellipse(n, rng):
    """
    Sample points uniformly inside unit ellipse u^2 + v^2 <= 1.
    """
    pts = []
    while len(pts) < n:
        cand = rng.uniform(-1.0, 1.0, size=(max(1000, n), 2))
        mask = np.sum(cand**2, axis=1) <= 1.0
        good = cand[mask]
        pts.extend(good.tolist())
    pts = np.array(pts[:n])
    return pts


def exponential_mu_targets(n_mu, R=20.0):
    """
    Desired fraction of fibers per MU.
    R = largest MU size / smallest MU size
    """
    b = R ** (1.0 / (n_mu - 1)) if n_mu > 1 else 1.0
    k = np.arange(n_mu)
    q = b ** k
    q = q / q.sum()
    return q


# ============================================================
# Fiber geometry generation
# ============================================================

def generate_fibers_3d(
    n_fibers=1500,
    n_nodes=80,
    muscle_length=12.0,
    a0=1.6,   # half-width in x at mid-belly
    b0=1.0,   # half-width in y at mid-belly
    taper=0.55,
    curvature_amp=0.25,
    pennation_amp_deg=10.0,
    seed=1
):
    """
    Generate synthetic 3D muscle fibers.

    Coordinate system:
      z: along muscle length
      x,y: cross-section coordinates

    The muscle is fusiform and mildly curved.
    Fibers follow the same general muscle axis and remain inside the muscle.
    """
    rng = np.random.default_rng(seed)

    # Fiber reference positions in normalized cross-section coordinates
    uv0 = sample_points_in_unit_ellipse(n_fibers, rng)   # each row in [-1,1], within unit disk/ellipse

    # Longitudinal coordinate
    z = np.linspace(0.0, muscle_length, n_nodes)
    zn = z / muscle_length  # normalized 0..1

    # Fusiform radius scaling along z (largest at mid-belly)
    belly = 1.0 - taper * (2.0 * np.abs(zn - 0.5))**1.5
    belly = np.clip(belly, 0.2, None)

    # Muscle centerline curvature
    x_center = curvature_amp * np.sin(2.0 * np.pi * zn)
    y_center = 0.08 * np.sin(np.pi * zn)

    # Pennation-like lateral drift along z
    theta = np.deg2rad(pennation_amp_deg) * np.sin(np.pi * zn)
    drift_x = 0.35 * np.tan(theta) * (zn - 0.5)
    drift_y = 0.10 * np.sin(2.0 * np.pi * zn)

    fibers = []
    midpoints = []

    for j in range(n_fibers):
        u0, v0 = uv0[j]

        # Each fiber occupies a similar normalized position through the muscle,
        # with a small fiber-specific waviness / drift
        phi = rng.uniform(0, 2*np.pi)
        local_wiggle_x = 0.03 * np.sin(4*np.pi*zn + phi)
        local_wiggle_y = 0.03 * np.cos(3*np.pi*zn + phi)

        x = x_center + (a0 * belly) * u0 + drift_x + local_wiggle_x
        y = y_center + (b0 * belly) * v0 + drift_y + local_wiggle_y
        zz = z.copy()

        fiber_xyz = np.column_stack([x, y, zz])
        fibers.append(fiber_xyz)

        mid_idx = n_nodes // 2
        midpoints.append([u0, v0, x[mid_idx], y[mid_idx], zz[mid_idx]])

    midpoints = np.array(midpoints)
    return fibers, midpoints, uv0


# ============================================================
# Motor unit assignment
# ============================================================

def assign_motor_units_3d_like_section34(
    uv0,
    n_mu=20,
    R=20.0,
    sigma_min=0.18,
    sigma_max=0.42,
    n_iter=250,
    seed=2
):
    """
    Assign each fiber to one MU using:
      - MU centers in normalized cross-section
      - radial basis function territory
      - lambda scaling to approximately enforce exponential MU size distribution

    uv0: (n_fibers,2), normalized cross-section coordinates for each fiber
    """
    rng = np.random.default_rng(seed)
    n_fibers = uv0.shape[0]

    # MU centers in [0,1)^2 from Weyl sequence -> remap to [-1,1]^2
    raw_centers = weyl_points(n_mu)
    centers = 2.0 * raw_centers - 1.0

    # Keep only centers that are inside unit ellipse-ish region by projection if needed
    for k in range(n_mu):
        r2 = np.sum(centers[k]**2)
        if r2 > 0.95**2:
            centers[k] *= 0.95 / np.sqrt(r2)

    # Desired exponential size fractions
    q = exponential_mu_targets(n_mu, R=R)

    # MU territory widths: larger MU -> broader territory
    sigmas = sigma_min + (sigma_max - sigma_min) * (
        (q - q.min()) / (q.max() - q.min() + 1e-12)
    )

    # Build raw RBF weights
    P = np.zeros((n_fibers, n_mu))
    for k in range(n_mu):
        d2 = np.sum((uv0 - centers[k])**2, axis=1)
        # Cauchy-like radial basis as in paper style
        a = np.pi**2 / (4.0 * sigmas[k]**4)
        P[:, k] = 1.0 / (1.0 + a * d2)

    # Solve for lambda factors iteratively
    lam = np.ones(n_mu)
    for _ in range(n_iter):
        W = P * lam[None, :]
        prob = W / W.sum(axis=1, keepdims=True)
        current = prob.mean(axis=0)
        lam *= q / np.maximum(current, 1e-12)

    W = P * lam[None, :]
    prob = W / W.sum(axis=1, keepdims=True)

    # Stochastic assignment
    mu_id = np.array([rng.choice(n_mu, p=prob[j]) for j in range(n_fibers)])

    return mu_id, centers, q, sigmas, prob


# ============================================================
# Visualization
# ============================================================

def plot_3d_fibers(
    fibers,
    mu_id,
    midpoints,
    centers_uv,
    highlight_mu=(0, 5, 10, 19),
    sample_gray=400,
    title="3D Motor Unit Distribution"
):
    """
    3D visualization:
      - many background fibers in light gray
      - highlighted MUs in color
    """
    n_fibers = len(fibers)
    rng = np.random.default_rng(123)
    gray_idx = rng.choice(n_fibers, size=min(sample_gray, n_fibers), replace=False)

    cmap = plt.colormaps.get_cmap("tab10")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Background fibers
    for idx in gray_idx:
        xyz = fibers[idx]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="lightgray", linewidth=0.5, alpha=0.35)

    # Highlighted MUs
    for cidx, k in enumerate(highlight_mu):
        ids = np.where(mu_id == k)[0]
        for idx in ids:
            xyz = fibers[idx]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=cmap(cidx), linewidth=1.0, alpha=0.85)

        # mark approximate MU center at mid-belly slice
        u, v = centers_uv[k]
        # just visualize center on z ~ mid-belly
        z0 = np.mean([fibers[0][0, 2], fibers[0][-1, 2]]) / 2.0
        ax.scatter(
            [u * 1.6], [v * 1.0], [z0],
            color=cmap(cidx), marker="x", s=100, linewidths=2,
            label=f"MU {k+1}"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_box_aspect((2.0, 1.4, 4.5))
    plt.tight_layout()
    plt.show()


def plot_midbelly_cross_section(midpoints, mu_id, centers_uv, highlight_mu=(0, 5, 10, 19)):
    """
    Plot normalized cross-section positions at mid-belly.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # all fibers
    ax.scatter(midpoints[:, 0], midpoints[:, 1], s=6, color="lightgray", alpha=0.35)

    cmap = plt.colormaps.get_cmap("tab10")
    for cidx, k in enumerate(highlight_mu):
        ids = np.where(mu_id == k)[0]
        ax.scatter(
            midpoints[ids, 0], midpoints[ids, 1],
            s=10, color=cmap(cidx), alpha=0.85, label=f"MU {k+1}"
        )
        ax.scatter(
            centers_uv[k, 0], centers_uv[k, 1],
            marker="x", s=120, linewidths=2, color=cmap(cidx)
        )

    # unit ellipse boundary (approx shown as unit circle in normalized coordinates)
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k--", linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlabel("normalized u")
    ax.set_ylabel("normalized v")
    ax.set_title("Mid-belly cross-section")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_mu_size_distribution(mu_id, n_mu, q_target=None):
    """
    Show realized counts vs target fractions.
    """
    counts = np.bincount(mu_id, minlength=n_mu)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(1, n_mu+1), counts, alpha=0.8, label="assigned fibers")

    if q_target is not None:
        target_counts = q_target * len(mu_id)
        ax.plot(np.arange(1, n_mu+1), target_counts, "r-", linewidth=2, label="target trend")

    ax.set_xlabel("Motor unit index")
    ax.set_ylabel("Fiber count")
    ax.set_title("Fiber count per MU")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return counts


# ============================================================
# Save results
# ============================================================

def save_assignment_csv(filename, fibers, mu_id):
    """
    Save one row per fiber with simplified summary.
    """
    rows = []
    for j, xyz in enumerate(fibers):
        mid = xyz[len(xyz)//2]
        start = xyz[0]
        end = xyz[-1]
        rows.append([
            j,
            mu_id[j],
            start[0], start[1], start[2],
            mid[0], mid[1], mid[2],
            end[0], end[1], end[2]
        ])

    arr = np.array(rows)
    header = (
        "fiber_id,mu_id,"
        "x_start,y_start,z_start,"
        "x_mid,y_mid,z_mid,"
        "x_end,y_end,z_end"
    )
    np.savetxt(filename, arr, delimiter=",", header=header, comments="")
    print(f"Saved: {filename}")


# ============================================================
# Main
# ============================================================

def main():
    # ---------------------------
    # Parameters
    # ---------------------------
    n_fibers = 1800
    n_nodes = 90
    n_mu = 20
    R = 18.0   # largest MU / smallest MU

    highlight_mu = (0, 4, 9, 19)

    # ---------------------------
    # Generate 3D fibers
    # ---------------------------
    fibers, midpoints, uv0 = generate_fibers_3d(
        n_fibers=n_fibers,
        n_nodes=n_nodes,
        muscle_length=12.0,
        a0=1.6,
        b0=1.0,
        taper=0.55,
        curvature_amp=0.20,
        pennation_amp_deg=9.0,
        seed=1
    )

    # ---------------------------
    # Assign motor units
    # ---------------------------
    mu_id, centers_uv, q_target, sigmas, prob = assign_motor_units_3d_like_section34(
        uv0,
        n_mu=n_mu,
        R=R,
        sigma_min=0.14,
        sigma_max=0.40,
        n_iter=250,
        seed=10
    )

    # ---------------------------
    # Visualizations
    # ---------------------------
    plot_3d_fibers(
        fibers,
        mu_id,
        midpoints,
        centers_uv,
        highlight_mu=highlight_mu,
        sample_gray=500,
        title="3D muscle fibers grouped into motor units"
    )

    plot_midbelly_cross_section(
        midpoints,
        mu_id,
        centers_uv,
        highlight_mu=highlight_mu
    )

    counts = plot_mu_size_distribution(mu_id, n_mu=n_mu, q_target=q_target)

    print("Counts per MU:")
    print(counts)
    print("Smallest MU fibers:", counts.min())
    print("Largest MU fibers :", counts.max())
    print("Ratio largest/smallest:", counts.max() / max(counts.min(), 1))

    # ---------------------------
    # Save summary table
    # ---------------------------
    save_assignment_csv("fiber_mu_assignment_3d.csv", fibers, mu_id)


if __name__ == "__main__":
    main()