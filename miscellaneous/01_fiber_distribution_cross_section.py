import numpy as np
import matplotlib.pyplot as plt

def weyl_points(n, seed=(np.sqrt(2), np.sqrt(3))):
    """
    2D Weyl low-discrepancy sequence in [0,1)^2.
    """
    i = np.arange(1, n + 1)[:, None]
    alpha = np.array(seed)[None, :]
    return (i * alpha) % 1.0


def generate_fibers_square(n_side=100):
    """
    Simple square cross-section prototype.
    Replace this later with points inside your real muscle mask.
    """
    xs = np.linspace(0.0, 1.0, n_side)
    ys = np.linspace(0.0, 1.0, n_side)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def exponential_mu_targets(n_mu, R=100.0):
    """
    Desired fraction of fibers per MU.
    R = largest MU size / smallest MU size.
    """
    b = R ** (1.0 / n_mu)
    k = np.arange(n_mu)
    q = b ** k
    q = q / q.sum()
    return q


def assign_motor_units(
    fiber_xy,
    n_mu=100,
    R=100.0,
    sigma_min=0.04,
    sigma_max=0.16,
    n_iter=200,
    seed=1,
):
    rng = np.random.default_rng(seed)
    n_fibers = fiber_xy.shape[0]

    # 1. MU centers
    centers = weyl_points(n_mu)

    # 2. Exponential desired MU sizes
    q = exponential_mu_targets(n_mu, R=R)

    # 3. Territory width: larger MUs get larger sigma
    sigmas = sigma_min + (sigma_max - sigma_min) * (
        (q - q.min()) / (q.max() - q.min())
    )

    # 4. Raw radial basis probabilities p_k(x_j)
    P = np.zeros((n_fibers, n_mu))
    for k in range(n_mu):
        d2 = np.sum((fiber_xy - centers[k]) ** 2, axis=1)
        a = np.pi**2 / (4.0 * sigmas[k] ** 4)
        P[:, k] = 1.0 / (1.0 + a * d2)

    # 5. Find lambda scaling factors so mean probabilities match q
    lam = np.ones(n_mu)

    for _ in range(n_iter):
        W = P * lam[None, :]
        prob = W / W.sum(axis=1, keepdims=True)
        current = prob.mean(axis=0)
        lam *= q / np.maximum(current, 1e-12)

    W = P * lam[None, :]
    prob = W / W.sum(axis=1, keepdims=True)

    # 6. Stochastic fiber-to-MU assignment
    mu_id = np.array([
        rng.choice(n_mu, p=prob[j])
        for j in range(n_fibers)
    ])

    return mu_id, centers, q, prob


def plot_assignment(fiber_xy, mu_id, centers, highlight=(0, 39, 99)):
    plt.figure(figsize=(6, 6))

    # Plot all fibers in light gray
    plt.scatter(fiber_xy[:, 0], fiber_xy[:, 1], s=2, alpha=0.15)

    # Highlight selected MUs
    for k in highlight:
        idx = mu_id == k
        plt.scatter(
            fiber_xy[idx, 0],
            fiber_xy[idx, 1],
            s=4,
            label=f"MU {k+1}"
        )
        plt.scatter(
            centers[k, 0],
            centers[k, 1],
            marker="x",
            s=120,
            linewidths=2,
        )

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_counts(mu_id, n_mu):
    counts = np.bincount(mu_id, minlength=n_mu)

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1, n_mu + 1), counts)
    plt.xlabel("Motor unit index")
    plt.ylabel("Number of fibers")
    plt.tight_layout()
    plt.show()

    return counts


if __name__ == "__main__":
    fiber_xy = generate_fibers_square(n_side=200)  # 40,000 fibers

    mu_id, centers, q, prob = assign_motor_units(
        fiber_xy,
        n_mu=100,
        R=30.0,
        sigma_min=0.035,
        sigma_max=0.14,
        seed=3,
    )

    plot_assignment(fiber_xy, mu_id, centers, highlight=(3, 39, 99))
    counts = plot_counts(mu_id, n_mu=100)

    # Save for downstream simulation
    output = np.column_stack([
        np.arange(len(fiber_xy)),
        fiber_xy[:, 0],
        fiber_xy[:, 1],
        mu_id
    ])

    np.savetxt(
        "fiber_mu_assignment.csv",
        output,
        delimiter=",",
        header="fiber_id,x,y,mu_id",
        comments=""
    )