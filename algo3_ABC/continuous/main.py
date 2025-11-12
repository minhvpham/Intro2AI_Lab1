import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path so we can import shared utilities and the
# ACO visualization functions used as a reference style.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.Continuous_functions import rastrigin
# Use the repository's ACOR visualization utilities (they include
# convergence and animated convergence helpers used by examples).
from algo1_ACO.rastrigin.visualization import (
    plot_convergence, plot_rastrigin_surface_2d, plot_animated_convergence,
    plot_animated_convergence_simple,
)


class ABC_Solver:
    """Artificial Bee Colony solver with archive tracking for visualization.

    The solver stores an "archive" equivalent (the population with costs) as
    an (N, D+1) array where the last column is the cost. This mirrors the
    ACOR solver structure used by the repository's reference implementation
    so the same visualization utilities can be reused.
    """

    def __init__(self, cost_function, n_dims, bounds,
                 pop_size=50, n_iterations=200, limit=20, track_archive=False):
        self.cost_function = cost_function
        self.n_dims = n_dims
        self.bounds_low, self.bounds_high = bounds
        self.pop = pop_size
        self.n_iterations = n_iterations
        self.limit = limit
        # whether to keep archive history for visualization (can be memory heavy)
        self.track_archive = bool(track_archive)
        # tuning parameters to improve accuracy
        self.selection_pressure = 2.0  # exponent on quality to sharpen selection
        self.local_search_steps = 3    # small greedy refinements when improvement found

        # population archive: (pop, n_dims + 1) -> [x..., cost]
        self.archive = np.zeros((self.pop, self.n_dims + 1))
        self.trial_counts = np.zeros(self.pop, dtype=int)

        self.best_solution = None
        self.best_cost = np.inf
        self.convergence_history = []
        self.archive_history = []

    def _initialize(self):
        # Use a simple Latin-hypercube like spread to initialize population for
        # better coverage of the search space which often improves optimizer accuracy.
        if self.pop > 1:
            # create evenly spaced samples per-dimension and shuffle per-dim
            cut = np.linspace(0, 1, self.pop + 1)
            rng = np.random.rand(self.pop, self.n_dims)
            pts = np.zeros((self.pop, self.n_dims))
            for d in range(self.n_dims):
                intervals = cut[:-1] + rng[:, d] * (cut[1] - cut[0])
                np.random.shuffle(intervals)
                pts[:, d] = intervals

            # map unit hypercube to bounds
            sols = self.bounds_low + pts * (self.bounds_high - self.bounds_low)
        else:
            sols = np.random.uniform(self.bounds_low, self.bounds_high, (1, self.n_dims))

        for i in range(self.pop):
            sol = sols[i]
            cost = self.cost_function(sol)
            self.archive[i, :-1] = sol
            self.archive[i, -1] = cost

        self._sort_archive()

    def _sort_archive(self):
        self.archive = self.archive[self.archive[:, -1].argsort()]
        if self.archive[0, -1] < self.best_cost:
            self.best_cost = self.archive[0, -1]
            self.best_solution = self.archive[0, :-1].copy()

    def run(self):
        np.random.seed(42)
        self._initialize()

        # record initial archive only when requested
        if self.track_archive and len(self.archive) > 0:
            self.archive_history.append(self.archive.copy())

        for gen in range(1, self.n_iterations + 1):
            # adaptive scale for neighborhood moves: shrink over time
            progress = (gen - 1) / max(1, self.n_iterations)
            move_scale = 0.5 * (1.0 - progress) + 0.05
            # Employed bees
            for i in range(self.pop):
                # select a partner different from i (if possible)
                if self.pop > 1:
                    choices = np.delete(np.arange(self.pop), i)
                    k = np.random.choice(choices)
                else:
                    k = i
                # sample a controlled perturbation; scale shrinks as iterations progress
                phi = np.random.uniform(-1, 1, self.n_dims) * move_scale
                xi = self.archive[i, :-1]
                xk = self.archive[k, :-1]
                v = xi + phi * (xi - xk)
                v = np.clip(v, self.bounds_low, self.bounds_high)
                v_cost = self.cost_function(v)
                if v_cost < self.archive[i, -1]:
                    self.archive[i, :-1] = v
                    self.archive[i, -1] = v_cost
                    self.trial_counts[i] = 0
                    # small local greedy refinement to reduce cost further
                    for _ in range(self.local_search_steps):
                        delta = np.random.normal(scale=0.2 * move_scale, size=self.n_dims) * (self.bounds_high - self.bounds_low)
                        cand = np.clip(self.archive[i, :-1] + delta, self.bounds_low, self.bounds_high)
                        cand_cost = self.cost_function(cand)
                        if cand_cost < self.archive[i, -1]:
                            self.archive[i, :-1] = cand
                            self.archive[i, -1] = cand_cost
                            # reset trial count after local improvement
                            self.trial_counts[i] = 0
                else:
                    self.trial_counts[i] += 1

            # Onlooker bees
            fitness = self.archive[:, -1]
            # convert fitness to selection probability (higher quality -> higher prob)
            quality = 1.0 / (1.0 + fitness + 1e-12)
            # apply selection pressure to focus on good solutions
            quality = np.power(quality, self.selection_pressure)
            total_q = np.sum(quality)
            if total_q <= 0 or not np.isfinite(total_q):
                prob = np.full(self.pop, 1.0 / max(1, self.pop))
            else:
                prob = quality / total_q

            for _ in range(self.pop):
                # select an index according to probability distribution
                try:
                    i = np.random.choice(self.pop, p=prob)
                except Exception:
                    # fallback to uniform selection
                    i = np.random.randint(0, self.pop)
                # choose a partner different from i when possible
                if self.pop > 1:
                    k_choices = np.delete(np.arange(self.pop), i)
                    k = np.random.choice(k_choices)
                else:
                    k = i
                phi = np.random.uniform(-1, 1, self.n_dims) * move_scale
                xi = self.archive[i, :-1]
                xk = self.archive[k, :-1]
                v = xi + phi * (xi - xk)
                v = np.clip(v, self.bounds_low, self.bounds_high)
                v_cost = self.cost_function(v)
                if v_cost < self.archive[i, -1]:
                    self.archive[i, :-1] = v
                    self.archive[i, -1] = v_cost
                    self.trial_counts[i] = 0
                    # local refinement
                    for _ in range(self.local_search_steps):
                        delta = np.random.normal(scale=0.2 * move_scale, size=self.n_dims) * (self.bounds_high - self.bounds_low)
                        cand = np.clip(self.archive[i, :-1] + delta, self.bounds_low, self.bounds_high)
                        cand_cost = self.cost_function(cand)
                        if cand_cost < self.archive[i, -1]:
                            self.archive[i, :-1] = cand
                            self.archive[i, -1] = cand_cost
                            self.trial_counts[i] = 0
                else:
                    self.trial_counts[i] += 1

            # Scout phase
            for i in range(self.pop):
                if self.trial_counts[i] > self.limit:
                    # scout: restart near current best solution (more effective than uniform random)
                    # get current best
                    best_pos = self.archive[0, :-1].copy()
                    # decreasing exploration radius
                    sigma = (0.5 * (1.0 - progress) + 0.01) * (self.bounds_high - self.bounds_low)
                    new_sol = np.clip(best_pos + np.random.normal(scale=sigma, size=self.n_dims), self.bounds_low, self.bounds_high)
                    self.archive[i, :-1] = new_sol
                    self.archive[i, -1] = self.cost_function(new_sol)
                    self.trial_counts[i] = 0

            # Update best and archive
            self._sort_archive()
            self.convergence_history.append(self.best_cost)
            if self.track_archive:
                self.archive_history.append(self.archive.copy())

            if gen % 50 == 0:
                print(f"Gen {gen}: Best Cost = {self.best_cost:.6e}")

        return self.best_solution, self.best_cost


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ABC Algorithm for Rastrigin Function Optimization")
    print("="*70)

    # Example 1: high-dimensional run (no archive tracking)
    print("\n[Example 1] Solving 5-D Rastrigin function...")
    DIMS_5D = 5
    BOUNDS = (-5.12, 5.12)
    POP = 50
    ITERS = 200

    abc5 = ABC_Solver(cost_function=rastrigin, n_dims=DIMS_5D, bounds=BOUNDS,
                      pop_size=POP, n_iterations=ITERS, track_archive=False)
    best5, cost5 = abc5.run()

    print("\n--- Results (5-D) ---")
    print(f"Best solution cost: {cost5:.6f}")
    print(f"Best solution vector: {best5}")
    print(f"Distance from global optimum (0): {np.linalg.norm(best5):.6f}")

    print("\nGenerating convergence plot...")
    plot_convergence(abc5.convergence_history)

    # Example 2: 2D run with visualization similar to ACOR reference
    print("\n" + "-"*70)
    print("[Example 2] Solving 2-D Rastrigin function with visualization...")
    DIMS_2D = 2
    ITERS_2D = 30

    print("\nVisualizing the Rastrigin function surface...")
    plot_rastrigin_surface_2d(bounds=BOUNDS)

    abc2 = ABC_Solver(cost_function=rastrigin, n_dims=DIMS_2D, bounds=BOUNDS,
                      pop_size=30, n_iterations=ITERS_2D, limit=20, track_archive=True)
    best2, cost2 = abc2.run()

    print("\n--- Results (2-D) ---")
    print(f"Best solution cost: {cost2:.6f}")
    print(f"Best solution vector: {best2}")
    print(f"Distance from global optimum (0,0): {np.linalg.norm(best2):.6f}")

    print("\nGenerating animated convergence with solution evolution (0.8s per iteration)...")
    plot_animated_convergence(
        history=abc2.convergence_history,
        archive_history=abc2.archive_history,
        bounds=BOUNDS,
        pause_time=0.8,
        cost_function_name="Rastrigin (2D)"
    )

    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)