import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import os


# =========================================
#  Rastrigin function
# =========================================
def rastrigin(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))


# =========================================
#  Generate Rastrigin surface
# =========================================
def rastrigin_surface(bounds=(-5.12, 5.12), resolution=200):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
    return X, Y, Z


# =========================================
#  Cuckoo Search (with history tracking)
# =========================================
def cuckoo_search_with_history(func, LB, UB, D=2, N=25, MaxGen=60, pa=0.25, alpha=0.01):
    np.random.seed(42)
    nests = np.random.uniform(LB, UB, size=(N, D))
    fitness = np.array([func(x) for x in nests])

    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_f = fitness[best_idx]

    history = [best_f]
    pop_history = [nests.copy()]

    # for gen in range(MaxGen):
    #     # Lévy-like step for all nests (including best)
    #     step = np.random.standard_normal(size=(N, D))
    #     new_nests = nests + alpha * step * np.random.standard_normal(size=(N, D))
    #     new_nests = np.clip(new_nests, LB, UB)

    #     new_fitness = np.array([func(x) for x in new_nests])

    #     # Replace if better
    #     better_mask = new_fitness < fitness
    #     nests[better_mask] = new_nests[better_mask]
    #     fitness[better_mask] = new_fitness[better_mask]

    #     # Abandon a fraction of worst nests
    #     n_abandon = int(pa * N)
    #     if n_abandon > 0:
    #         worst_idx = np.argsort(fitness)[-n_abandon:]
    #         nests[worst_idx] = np.random.uniform(LB, UB, size=(n_abandon, D))
    #         fitness[worst_idx] = np.array([func(x) for x in nests[worst_idx]])

    #     # Update global best
    #     best_idx = np.argmin(fitness)
    #     if fitness[best_idx] < best_f:
    #         best_f = fitness[best_idx]
    #         best_nest = nests[best_idx].copy()

    #     history.append(best_f)
    #     pop_history.append(nests.copy())
    for gen in range(MaxGen):
        # adaptive step size
        alpha_t = 0.01 * (UB - LB) * (0.97 ** gen)
        beta = 0.8

        # Lévy-like exploration + attraction to best
        step = np.random.standard_normal(size=(N, D))
        new_nests = nests + alpha_t * step * np.random.standard_normal(size=(N, D)) \
                    + beta * (best_nest - nests)
        new_nests = np.clip(new_nests, LB, UB)

        new_fitness = np.array([func(x) for x in new_nests])

        # Greedy selection
        better = new_fitness < fitness
        nests[better] = new_nests[better]
        fitness[better] = new_fitness[better]

        # Abandon a few worst nests
        n_abandon = int(pa * N)
        if n_abandon > 0:
            worst_idx = np.argsort(fitness)[-n_abandon:]
            nests[worst_idx] = np.random.uniform(LB, UB, size=(n_abandon, D))
            fitness[worst_idx] = [func(x) for x in nests[worst_idx]]

        # Update best
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_f:
            best_f = fitness[cur_best_idx]
            best_nest = nests[cur_best_idx].copy()

        history.append(best_f)
        pop_history.append(nests.copy())

    return best_nest, best_f, history, pop_history


# =========================================
#  Visualization with movement arrows (saved in GIF)
# =========================================
def visualize_cuckoo_search(pop_history, history, bounds=(-5.12, 5.12),
                             filename="results/cuckoo_rastrigin.gif", fps=8):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    X, Y, Z = rastrigin_surface(bounds)
    frames = []

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = cm.viridis
    trails = []

    for gen, nests in enumerate(pop_history):
        ax.clear()
        ax.contourf(X, Y, Z, levels=60, cmap=cmap)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)

        # Plot nests
        ax.scatter(nests[:, 0], nests[:, 1], color="dodgerblue", s=35, label="Nests")

        # Global minimum (for reference)
        ax.scatter(0, 0, c='white', s=100, marker='X', edgecolor='black', label="Global Minimum")

        # Best nest and trail
        best_idx = np.argmin([rastrigin(x) for x in nests])
        best_nest = nests[best_idx]
        trails.append(best_nest)
        ax.scatter(best_nest[0], best_nest[1], c='red', s=120, marker='*', label="Best Nest")

        if len(trails) > 1:
            t = np.array(trails)
            ax.plot(t[:, 0], t[:, 1], 'r--', lw=1.5, alpha=0.7)

        # Draw movement arrows for *all* nests
        if gen > 0:
            prev = pop_history[gen - 1]
            for i in range(len(nests)):
                dx, dy = nests[i, 0] - prev[i, 0], nests[i, 1] - prev[i, 1]
                ax.arrow(
                    prev[i, 0], prev[i, 1], dx, dy,
                    color='orange', alpha=0.6, width=0.005, head_width=0.15,
                    length_includes_head=True
                )

        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_title(f"Cuckoo Search - Iter {gen+1:03d}\nBest Cost: {history[gen]:.4f}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

        # Save frame
        frame_path = f"results/_frame_{gen:03d}.png"
        plt.savefig(frame_path, dpi=120)
        frames.append(imageio.imread(frame_path))

    # Save GIF (with arrows)
    imageio.mimsave(filename, frames, fps=fps)
    print(f"✅ GIF saved to: {filename}")

    # Cleanup
    for gen in range(len(pop_history)):
        try:
            os.remove(f"results/_frame_{gen:03d}.png")
        except:
            pass


# =========================================
#  Run demo
# =========================================
if __name__ == "__main__":
    print("🎬 Running full Cuckoo Search visualization with movement arrows...")
    LB, UB = -15.12, 15.12

    best, best_f, hist, pop_hist = cuckoo_search_with_history(
        func=rastrigin, LB=LB, UB=UB, D=2, N=30, MaxGen=60, pa=0.25
    )

    visualize_cuckoo_search(
        pop_hist, hist, bounds=(LB, UB),
        filename="results/cuckoo_rastrigin.gif", fps=8
    )
