import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ---------------------- Problem Definition (TSP) ---------------------- #
np.random.seed(42)
NUM_CITIES = 20

# ---------------------- ABC Algorithm Parameters ---------------------- #
N = 40              # Number of food sources (population size)
D = NUM_CITIES      # Dimension (one per city)
MaxGen = 1000       # Number of iterations
limit = 40          # Trial limit for scout bees
# ---------------------------------------------------------------------- #


def generate_cities(n):
    """Generate random (x,y) coordinates in [0,100]."""
    return np.random.rand(n, 2) * 100

def get_distance_matrix(cities):
    """Vectorized Euclidean distance matrix."""
    diff = cities[:, None, :] - cities[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

# --- Fitness Function ---

def calculate_path_distance(path, dist_matrix):
    """Compute the total path distance for one permutation."""
    # Use np.roll to efficiently get the 'next' city, including wrap-around
    return np.sum(dist_matrix[path, np.roll(path, -1)])

# --- "Movement" Operators (Local Search) ---

def two_opt_mutation(path):
    """
    Perform a *single random* 2-opt move (reverse a segment).
    Used by Employed and Onlooker bees for fast exploration.
    """
    i, j = np.sort(np.random.choice(len(path), 2, replace=False))
    new_path = path.copy()
    new_path[i:j+1] = new_path[i:j+1][::-1]
    return new_path

# --- NEW: Powerful Greedy 2-Opt Local Search ---
def greedy_two_opt(path, dist_matrix):
    """
    Performs a full, greedy 2-Opt local search.
    Repeatedly iterates and applies 2-Opt swaps until no
    further improvement is possible (local optimum).
    
    This is the "memetic" part of the algorithm.
    """
    best_path = path.copy()
    best_dist = calculate_path_distance(best_path, dist_matrix)
    n = len(best_path)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):       # Iterate over all possible start points
            for j in range(i + 1, n): # Iterate over all possible end points
                
                # Create the new path by reversing the segment [i:j+1]
                new_path = best_path.copy()
                new_path[i:j+1] = new_path[i:j+1][::-1]
                
                new_dist = calculate_path_distance(new_path, dist_matrix)
                
                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True # An improvement was made, re-scan
                    # In a "first improvement" strategy, we'd break here.
                    # In a "best improvement" strategy, we'd continue the inner loops.
                    # For simplicity and speed, we'll use this "first improvement" style.
                    break # Break inner loop
            if improved:
                break # Break outer loop
                
    return best_path, best_dist


# --- Scout Bee Operator ---

def nearest_neighbor_path(dist_matrix, D):
    """
    "Smarter Scout": Creates a new, high-quality path
    using the Nearest Neighbor greedy heuristic.
    """
    start_city = np.random.randint(D)
    path = [start_city]
    unvisited = set(range(D))
    unvisited.remove(start_city)
    
    current_city = start_city
    while unvisited:
        # Find the nearest unvisited city
        distances = dist_matrix[current_city]
        
        # Filter distances to only include unvisited cities
        valid_distances = [(dist, city) for city, dist in enumerate(distances) if city in unvisited]
        
        # Find the minimum distance
        next_city = min(valid_distances, key=lambda x: x[0])[1]
        
        path.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
        
    return np.array(path)

# ---------------- ABC core (Memetic) ---------------- #

def artificial_bee_colony_tsp(dist_matrix, D, N, MaxGen, limit):
    
    # Initialize
    foods = np.array([np.random.permutation(D) for _ in range(N)])
    fitness = np.array([calculate_path_distance(p, dist_matrix) for p in foods])
    trial = np.zeros(N, dtype=int)

    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_path = foods[best_idx].copy()
    history = [best_fit]

    print(f"Initial Best Distance: {best_fit:.2f}")

    for gen in range(1, MaxGen + 1):
        # ---------- Employed bees (Exploration) ----------
        for i in range(N):
            new_path = two_opt_mutation(foods[i])
            new_fit = calculate_path_distance(new_path, dist_matrix)
            if new_fit < fitness[i]:
                foods[i], fitness[i], trial[i] = new_path, new_fit, 0
            else:
                trial[i] += 1

        # ---------- Onlooker bees (Exploitation) ----------
        inv_fit = 1.0 / (1.0 + fitness)
        prob = inv_fit / np.sum(inv_fit)
        for _ in range(N):
            i = np.random.choice(N, p=prob)
            new_path = two_opt_mutation(foods[i])
            new_fit = calculate_path_distance(new_path, dist_matrix)
            if new_fit < fitness[i]:
                foods[i], fitness[i], trial[i] = new_path, new_fit, 0
            else:
                trial[i] += 1
                
        # --- NEW: Memetic Local Search Phase (Intensification) ---
        # Apply the full, greedy 2-Opt search to the *current best*
        current_best_idx = np.argmin(fitness)
        # We only run this if the best solution is new this gen
        if trial[current_best_idx] == 0:
            opt_path, opt_fit = greedy_two_opt(foods[current_best_idx], dist_matrix)
            # If the greedy search found an even better solution, update it
            if opt_fit < fitness[current_best_idx]:
                foods[current_best_idx] = opt_path
                fitness[current_best_idx] = opt_fit
                # trial[current_best_idx] is already 0

        # ---------- Scout bees (Diversification) ----------
        scouts = np.where(trial > limit)[0]
        if scouts.size > 0:
            for i in scouts:
                # Use "Smarter Scout"
                foods[i] = nearest_neighbor_path(dist_matrix, D)
                fitness[i] = calculate_path_distance(foods[i], dist_matrix)
                trial[i] = 0

        # ---------- Global best ----------
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fit:
            best_fit = fitness[min_idx]
            best_path = foods[min_idx].copy()

        if gen % 100 == 0:
            print(f"Gen {gen}: Best Distance = {best_fit:.2f}")

        history.append(best_fit)

    return best_path, best_fit, history

# ---------------- Plotting ---------------- #
def plot_tsp_solution(cities, path, title="TSP Solution"):
    path_coords = cities[path]
    path_coords = np.vstack([path_coords, path_coords[0]])
    plt.figure(figsize=(10, 8))
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'o-', lw=2)
    plt.scatter(cities[:, 0], cities[:, 1], color='red', s=80)
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.8, y + 0.8, str(i), fontsize=10)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsp_abc_memetic.png", dpi=200)
    print("\n✅ Saved plot to 'tsp_abc_memetic.png'")

# ---------------- Run ---------------- #
if __name__ == "__main__":
    print(f"--- Memetic ABC for TSP ({NUM_CITIES} cities) ---")
    cities = generate_cities(NUM_CITIES)
    dist_matrix = get_distance_matrix(cities)
    best_path, best_fit, hist = artificial_bee_colony_tsp(
        dist_matrix, NUM_CITIES, N, MaxGen, limit
    )

    print("\n--- ✅ Memetic Optimization Complete ---")
    print("Best Path:", best_path)
    print(f"Best Distance: {best_fit:.2f}")
    plot_tsp_solution(cities, best_path, f"Memetic ABC-TSP ({NUM_CITIES} cities)\nDistance = {best_fit:.2f}")