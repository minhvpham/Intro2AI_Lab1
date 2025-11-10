import numpy as np
import math
from typing import Callable, Tuple, List

# ---------------------- Problem & Algorithm Config ---------------------- #
# choose the function and bounds before calling cuckoo_search
# e.g. func_to_optimize = rastrigin_function; LB, UB = -5.12, 5.12
# ----------------------------------------------------------------------- #

def levy_flights_batch(N: int, D: int, Lambda: float = 1.5) -> np.ndarray:
    """
    Vectorized generation of N Lévy-flight step vectors of dimension D
    using Mantegna's algorithm. Returns shape (N, D).
    """
    # Mantegna's sigma for u
    sigma_u = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2.0) /
               (math.gamma((1 + Lambda) / 2.0) * Lambda * 2.0 ** ((Lambda - 1.0) / 2.0))) ** (1.0 / Lambda)
    
    u = np.random.normal(0.0, sigma_u, size=(N, D))
    v = np.random.normal(0.0, 1.0, size=(N, D))
    steps = u / (np.abs(v) ** (1.0 / Lambda) + 1e-16) # Add epsilon to avoid divide by zero
    return steps


def batch_evaluate(func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    """
    Evaluate objective function on a batch of solutions X (shape (N, D)).
    Returns 1-D array of length N.
    """
    # This loop is kept as not all objective functions are guaranteed
    # to be fully vectorized (e.g., Rosenbrock is tricky).
    # This is a safe and robust way to handle it.
    return np.array([func(x) for x in X])


def cuckoo_search(
    func_to_optimize: Callable[[np.ndarray], float],
    LB: float,
    UB: float,
    D: int,
    N: int = 50,
    MaxGen: int = 1000,
    pa: float = 0.25,
    alpha: float = None,
    Lambda: float = 1.5
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Optimized Cuckoo Search (vectorized) for continuous problems.
    This version implements "Modified Cuckoo Search" (MCS) for faster
    convergence by using a greedy selection in the global search phase.

    Parameters
    ----------
    func_to_optimize : callable
        Objective function (takes 1D numpy array of length D, returns scalar).
    LB, UB : float
        Lower and upper bound (same for all dimensions).
    D : int
        Dimensionality.
    N : int
        Number of nests (population size).
    MaxGen : int
        Number of iterations (generations).
    pa : float
        Fraction of nests to abandon each generation (0..1).
    alpha : float or None
        Step size scale. If None, set to 0.01*(UB-LB).
    Lambda : float
        Lévy exponent (typical 1.5).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    best_nest : np.ndarray
        Best solution found (length D).
    best_f : float
        Best objective value (lower is better).
    history : list
        Best fitness history per generation (len = MaxGen+1).
    """
    np.random.seed(42)
    if alpha is None:
        alpha = 0.01 * (UB - LB)

    # Initialize nests uniformly
    nests = np.random.uniform(LB, UB, size=(N, D))
    fitness = batch_evaluate(func_to_optimize, nests)

    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_f = float(fitness[best_idx])

    history = [best_f]

    n_abandon = max(1, int(round(pa * N)))  # ensure at least one when pa>0

    for gen in range(1, MaxGen + 1):
        # ----------------- Generate new cuckoo solutions (vectorized) -----------------
        steps = levy_flights_batch(N, D, Lambda=Lambda)        # (N, D)
        candidates = nests + alpha * steps
        np.clip(candidates, LB, UB, out=candidates)

        cand_fitness = batch_evaluate(func_to_optimize, candidates)  # (N,)

        # --- "FIX" / OPTIMIZATION: Modified Greedy Selection ---
        # Compare candidate 'i' directly with its parent 'i'.
        # This is the "Modified Cuckoo Search" (MCS) optimization.
        # It's algorithmically better and simpler to write.
        mask = cand_fitness < fitness  # boolean mask length N

        # Replace those nests where candidate is better
        # This is a direct, efficient boolean mask assignment
        nests[mask] = candidates[mask]
        fitness[mask] = cand_fitness[mask]

        # ----------------- Abandon worst nests and local search replace -----------------
        # identify worst indices (largest fitness)
        worst_idx = np.argsort(fitness)[-n_abandon:]

        # Create local candidate solutions for worst nests using differential-style move
        if worst_idx.size > 0:
            # For each worst index i, pick two other distinct indices j,k
            JK = np.empty((len(worst_idx), 2), dtype=int)
            for t, i_w in enumerate(worst_idx):
                pool = np.delete(np.arange(N), i_w)
                JK[t] = np.random.choice(pool, size=2, replace=False)

            j_idx = JK[:, 0]
            k_idx = JK[:, 1]
            r = np.random.rand(len(worst_idx), 1) # Random scale factor per abandoned nest

            new_local = nests[worst_idx] + r * (nests[j_idx] - nests[k_idx])
            np.clip(new_local, LB, UB, out=new_local)
            new_local_f = batch_evaluate(func_to_optimize, new_local)

            # Replace only when new local is better
            replace_mask = new_local_f < fitness[worst_idx]
            
            # Note: We must index worst_idx with the selection 'sel'
            sel = np.where(replace_mask)[0] 
            if sel.size > 0:
                nests[worst_idx[sel]] = new_local[sel]
                fitness[worst_idx[sel]] = new_local_f[sel]

        # ----------------- Update best -----------------
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_f:
            best_f = float(fitness[cur_best_idx])
            best_nest = nests[cur_best_idx].copy()

        history.append(best_f)

        # optional logging
        if gen % max(1, (MaxGen // 10)) == 0:
            print(f"Gen {gen}/{MaxGen}: Best = {best_f:.6e}")

    return best_nest, best_f, history


# ---------------- Example: benchmark functions ---------------- #
def sphere_function(x: np.ndarray) -> float:
    return float(np.sum(x * x))

def rastrigin_function(x: np.ndarray) -> float:
    A = 10.0
    return float(A * len(x) + np.sum(x * x - A * np.cos(2 * math.pi * x)))

def rosenbrock_function(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0))

def ackley_function(x: np.ndarray) -> float:
    n = len(x)
    sum1 = np.sum(x * x)
    sum2 = np.sum(np.cos(2.0 * math.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return float(term1 + term2 + 20.0 + math.e)


# ---------------- If run as script: simple test ---------------- #
if __name__ == "__main__":
    # choose function and bounds
    func = rastrigin_function
    LB, UB = -5.12, 5.12
    D = 30
    N = 50
    MaxGen = 500
    pa = 0.25

    print(f"--- Running Optimized-Vectorized CS for: {func.__name__} ---")
    print(f"Bounds: [{LB}, {UB}], Dimensions: {D}, Population: {N}, Generations: {MaxGen}")
    
    best_sol, best_val, hist = cuckoo_search(func, LB, UB, D, N=N, MaxGen=MaxGen, pa=pa, seed=42)

    print("\n--- ✅ Optimization Complete ---")
    print(f"Final best objective: {best_val:.6e}")
    # print("First 8 dims of best sol:", best_sol[:8])