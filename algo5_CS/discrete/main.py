import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ---------------------- Problem Definition (TSP) ---------------------- #
np.random.seed(42)
NUM_CITIES = 20

# ---------------------- CS Algorithm Parameters ---------------------- #
N = 40              # Number of nests (population size)
D = NUM_CITIES      # Dimension (one per city)
MaxGen = 1000       # Number of iterations
pa = 0.25           # Discovery rate (fraction of nests to abandon)
# --------------------------------------------------------------------- #


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

# --- Cuckoo Search Operators ---

def exploration_2opt(path):
    """
    "Global Search" operator (replaces Lévy flight).
    Performs a *single random* 2-opt move (reverse a segment).
    """
    i, j = np.sort(np.random.choice(len(path), 2, replace=False))
    new_path = path.copy()
    new_path[i:j+1] = new_path[i:j+1][::-1]
    return new_path

# --- "Smarter Scout" Operator ---

def nearest_neighbor_path(dist_matrix, D):
    """
    "Smarter Scout": Creates a new, high-quality path
    using the Nearest Neighbor greedy heuristic.
    This replaces the simple 'local_search_2opt' in the abandonment phase.
    """
    start_city = np.random.randint(D)
    path = [start_city]
    unvisited = set(range(D))
    unvisited.remove(start_city)
    
    current_city = start_city
    while unvisited:
        distances = dist_matrix[current_city]
        valid_distances = [(dist, city) for city, dist in enumerate(distances) if city in unvisited]
        next_city = min(valid_distances, key=lambda x: x[0])[1]
        
        path.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
        
    return np.array(path)

# --- NEW: Powerful Greedy 2-Opt Local Search (Memetic Step) ---

def greedy_two_opt(path, dist_matrix):
    """
    Performs a full, greedy 2-Opt local search.
    Repeatedly iterates and applies 2-Opt swaps until no
    further improvement is possible (local optimum).
    """
    best_path = path.copy()
    best_dist = calculate_path_distance(best_path, dist_matrix)
    n = len(best_path)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_path = best_path.copy()
                new_path[i:j+1] = new_path[i:j+1][::-1]
                new_dist = calculate_path_distance(new_path, dist_matrix)
                
                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True
                    break 
            if improved:
                break
                
    return best_path, best_dist

# --- Main Memetic Cuckoo Search Algorithm ---

def cuckoo_search_tsp(dist_matrix, D, N, MaxGen, pa):
    """
    Main Memetic CS algorithm adapted for the TSP (minimization).
    
    Returns:
        tuple: (best_path, best_distance, history_of_best_distance)
    """
    
    np.random.seed(42)

    # Step 1: Initialize nests (as permutations)
    nests = np.array([np.random.permutation(D) for _ in range(N)])
    fitness = np.array([calculate_path_distance(p, dist_matrix) for p in nests])
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_path = nests[best_idx].copy()
    
    history_list = [best_fitness]
    
    print(f"Initial Best Distance: {best_fitness:.2f}")

    for gen in range(1, MaxGen + 1):
        
        # Step 2: Generate new solutions (Global Search)
        for i in range(N):
            # Use 2-Opt as the exploration "flight"
            new_path = exploration_2opt(nests[i])
            new_f = calculate_path_distance(new_path, dist_matrix)
            
            # Step 3: Greedy selection
            j = np.random.randint(0, N)
            if new_f < fitness[j]:
                nests[j] = new_path
                fitness[j] = new_f

        # Step 4: Abandon worst nests (Diversification)
        n_abandon = int(pa * N)
        if n_abandon > 0:
            worst_indices = np.argsort(fitness)[-n_abandon:]
            
            for i in worst_indices:
                # OPTIMIZATION 1: Use "Smarter Scout"
                nests[i] = nearest_neighbor_path(dist_matrix, D)
                fitness[i] = calculate_path_distance(nests[i], dist_matrix)
        
        # OPTIMIZATION 2: Memetic Step (Intensification)
        # Apply greedy 2-Opt to the current best solution
        current_best_idx = np.argmin(fitness)
        opt_path, opt_fit = greedy_two_opt(nests[current_best_idx], dist_matrix)
        
        # Update the population if the greedy search found an improvement
        if opt_fit < fitness[current_best_idx]:
            nests[current_best_idx] = opt_path
            fitness[current_best_idx] = opt_fit

        # Step 5: Update global best
        current_best_idx = np.argmin(fitness) # Re-check in case memetic step changed it
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_path = nests[current_best_idx].copy()
            
        if gen % 100 == 0:
            print(f"Gen {gen}: Best Distance = {best_fitness:.2f}")
            
        history_list.append(best_fitness)
            
    return best_path, best_fitness, history_list

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
    plt.savefig("tsp_cuckoo_memetic.png", dpi=200)
    print("\n✅ Saved plot to 'tsp_cuckoo_memetic.png'")

# ---------------- Run ---------------- #
if __name__ == "__main__":
    print(f"--- Memetic Cuckoo Search for TSP ({NUM_CITIES} cities) ---")
    cities = generate_cities(NUM_CITIES)
    dist_matrix = get_distance_matrix(cities)
    
    best_path, best_fit, hist = cuckoo_search_tsp(
        dist_matrix, NUM_CITIES, N, MaxGen, pa
    )

    print("\n--- ✅ Memetic Optimization Complete ---")
    print("Best Path:", best_path)
    print(f"Best Distance: {best_fit:.2f}")
    plot_tsp_solution(cities, best_path, f"Memetic CS-TSP ({NUM_CITIES} cities)\nDistance = {best_fit:.2f}")