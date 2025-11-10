import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import os
from main import cuckoo_search, rastrigin_function  # import your code


# ---------------------- Generate Surface ---------------------- #
def create_rastrigin_surface(bounds=(-5.12, 5.12), resolution=200):
    """Create a 2D grid of the Rastrigin function."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
    return X, Y, Z


# ---------------------- Run CS (2D + History) ---------------------- #
def run_cuckoo_with_history(func, LB, UB, D=2, N=30, MaxGen=50, pa=0.25):
    """Modified cuckoo_search that records population history for visualization."""
    np.random.seed(0)
    alpha = 0.01 * (UB - LB)
    nests = np.random.uniform(LB, UB, size=(N, D))
    fitness = np.array([func(x) for x in nests])

    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_f = fitness[best_idx]

    history = [best_f]
    pop_history = [nests.copy()]

    for gen in range(MaxGen):
        steps = np.random.standard_normal(size=(N, D))
        candidates = nests + alpha * steps
        np.clip(candidates, LB, UB, out=candidates)
        cand_fitness = np.array([func(x) for x in candidates])

        mask = cand_fitness < fitness
        nests[mask] = candidates[mask]
        fitness[mask] = cand_fitness[mask]

        # abandon fraction pa
        n_abandon = int(pa * N)
        if n_abandon > 0:
            worst_idx = np.argsort(fitness)[-n_abandon:]
            nests[worst_idx] = np.random.uniform(LB, UB, size=(n_abandon, D))
            fitness[worst_idx] = np.array([func(x) for x in nests[worst_idx]])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_f:
            best_nest = nests[best_idx].copy()
            best_f = fitness[best_idx]

        history.append(best_f)
        pop_history.append(nests.copy())

    return best_nest, best_f, history, pop_history


# ---------------------- Animate ---------------------- #
def make_cuckoo_animation(pop_history, history, bounds=(-5.12, 5.12),
                          filename="results/cuckoo_rastrigin.gif", fps=6):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    X, Y, Z = create_rastrigin_surface(bounds)
    global_min = np.array([0, 0])
    frames = []

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = cm.viridis
    trails = []

    for gen, nests in enumerate(pop_history):
        ax.clear()
        contour = ax.contourf(X, Y, Z, levels=60, cmap=cmap)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        if gen == 0:
            plt.colorbar(contour, ax=ax, shrink=0.8)

        # Find best nest
        best_idx = np.argmin([rastrigin_function(x) for x in nests])
        best_nest = nests[best_idx]
        trails.append(best_nest)

        # Plot nests
        ax.scatter(nests[:, 0], nests[:, 1], c='dodgerblue', s=30, label="Nests")

        # Global best (so far)
        ax.scatter(best_nest[0], best_nest[1], c='red', s=120, marker='*', label="Best Nest")

        # Global minimum
        ax.scatter(global_min[0], global_min[1], c='white', s=100, marker='X', label="Global Minimum")

        # Draw best trail
        if len(trails) > 1:
            trail = np.array(trails)
            ax.plot(trail[:, 0], trail[:, 1], 'r--', lw=1.5, alpha=0.7, label="Best Path" if gen == 0 else "")

        ax.set_title(f"Iteration {gen+1:03d}\nBest Cost: {history[gen]:.4f}", fontsize=11)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.legend(fontsize=8, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

        frame_path = f"results/_frame_{gen:03d}.png"
        plt.savefig(frame_path, dpi=120)
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(filename, frames, fps=fps)
    print(f"✅ GIF saved to {filename}")

    # Clean up
    for f in [f"results/_frame_{gen:03d}.png" for gen in range(len(pop_history))]:
        try:
            os.remove(f)
        except:
            pass


# ---------------------- Run Visualization ---------------------- #
if __name__ == "__main__":
    print("🎬 Running Cuckoo Search Visualization on Rastrigin 2D")
    func = rastrigin_function
    LB, UB = -5.12, 5.12

    best_nest, best_f, hist, pop_hist = run_cuckoo_with_history(
        func, LB, UB, D=2, N=25, MaxGen=60, pa=0.25
    )

    make_cuckoo_animation(pop_hist, hist, bounds=(LB, UB), filename="results/cuckoo_rastrigin.gif", fps=8)
