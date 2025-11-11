import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import os


# ---------------------- Objective Function ---------------------- #
def rastrigin(X):
    """Rastrigin function for 2D or ND input"""
    A = 10
    X = np.asarray(X)
    return A * X.size + np.sum(X**2 - A * np.cos(2 * np.pi * X))


# ---------------------- Surface ---------------------- #
def create_rastrigin_surface(bounds=(-5.12, 5.12), resolution=200):
    """Create a 2D grid of the Rastrigin function"""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    # Compute Z for each (x, y)
    Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
    return X, Y, Z


# ---------------------- Animated GIF ---------------------- #
def make_abc_animation(
    archive_history,
    history,
    bounds=(-5.12, 5.12),
    filename="results/abc_rastrigin.gif",
    fps=8,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    X, Y, Z = create_rastrigin_surface(bounds)
    global_min = np.array([0, 0])  # known for Rastrigin
    n_frames = min(len(history), len(archive_history))
    frame_step = max(1, n_frames // 60)

    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = cm.viridis

    bee_trails = []  # store paths of best bees

    for gen in range(0, n_frames, frame_step):
        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        ax.contour(X, Y, Z, levels=20, colors="black", alpha=0.3, linewidths=0.5)

        archive = np.array(archive_history[gen])
        best_idx = np.argmin([rastrigin(a) for a in archive])
        best_bee = archive[best_idx]
        bee_trails.append(best_bee)

        # 🐝 Draw all bees
        ax.scatter(archive[:, 0], archive[:, 1], c="dodgerblue", s=30, label="Bees")

        # 🔴 Global best
        ax.scatter(best_bee[0], best_bee[1], c="red", s=120, marker="*", label="Best Bee")

        # 🟢 Global minimum (known)
        ax.scatter(global_min[0], global_min[1], c="white", s=100, marker="X", label="Global Minimum")

        # 🧭 Trail of best bee
        if len(bee_trails) > 1:
            trail = np.array(bee_trails)
            ax.plot(trail[:, 0], trail[:, 1], "r--", lw=1.5, alpha=0.7, label="Best Path" if gen == 0 else "")

        ax.set_title(f"Iteration {gen+1:03d}\nBest Cost: {history[gen]:.4f}", fontsize=11)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.legend(fontsize=8, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

        frame_path = f"results/_frame_{gen:03d}.png"
        plt.savefig(frame_path, dpi=120)
        frames.append(imageio.imread(frame_path))

    # 🎞 Save GIF
    imageio.mimsave(filename, frames, fps=fps)
    print(f"✅ Animated GIF saved to: {filename}")

    # 🧹 Clean up
    for f in [f"results/_frame_{gen:03d}.png" for gen in range(0, n_frames, frame_step)]:
        try:
            os.remove(f)
        except:
            pass


# ---------------------- Example Run ---------------------- #
if __name__ == "__main__":
    from main import ABC_Solver  # adjust import

    BOUNDS = (-5.12, 5.12)
    abc = ABC_Solver(
        cost_function=rastrigin,
        n_dims=2,
        bounds=BOUNDS,
        pop_size=30,
        n_iterations=80,
        limit=20,
        track_archive=True,
    )

    best_sol, best_cost = abc.run()

    make_abc_animation(
        archive_history=abc.archive_history,
        history=abc.convergence_history,
        bounds=BOUNDS,
        filename="results/abc_rastrigin.gif",
        fps=6,
    )
