import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.tsp import create_cities

def calculate_distance_matrix(cities):
    """
    Calculates the N x N pairwise Euclidean distance matrix
    from a (N, D) city coordinate matrix, using only NumPy. [33]
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 0)
    return np.sqrt(dist_sq)

def calculate_tour_cost(tour, dist_matrix):
    """
    Calculates the total cost (fitness) of a tour (permutation)
    using the pre-calculated distance matrix. 
    """
    from_cities = tour
    to_cities = np.roll(tour, -1)
    segment_costs = dist_matrix[from_cities, to_cities]
    return np.sum(segment_costs)

# --- Section III: "Calculus of Swaps" Operators ---

def subtract_permutations(p_target, p_current):
    """
    Implements the discrete subtraction operator: V = p_target - p_current.
    Generates a 'velocity' (list of swaps) that transforms
    p_current into p_target. 
    """
    velocity = []
    p_temp = p_current.copy()
    value_to_index_map = {city: i for i, city in enumerate(p_temp)}
    
    for i in range(len(p_temp)):
        if p_temp[i]!= p_target[i]:
            target_city = p_target[i]
            j = value_to_index_map[target_city]
            city_at_i = p_temp[i]
            
            p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
            value_to_index_map[target_city] = i
            value_to_index_map[city_at_i] = j
            velocity.append((i, j))
            
    return velocity

def multiply_velocity(c, velocity):
    """
    Implements the discrete scalar multiplication operator: V_scaled = c * V.
    'c' is a probability that determines if a swap is kept. 
    """
    mask = np.random.rand(len(velocity)) < c
    return [swap for i, swap in enumerate(velocity) if mask[i]]

def add_velocities(v1, v2):
    """
    Implements the discrete addition operator: V_new = v1 + v2.
    Combines two velocities (swap lists) by concatenation. 
    """
    return v1 + v2

def apply_velocity_to_position(position, velocity):
    """
    Implements the discrete position update: X_new = X + V.
    Applies a velocity (list of swaps) to a position (permutation).
    """
    new_position = position.copy()
    for i, j in velocity:
        new_position[i], new_position[j] = new_position[j], new_position[i]
    return new_position

# --- Section IV: 2-Opt Local Search ---

def apply_2_opt(tour, dist_matrix):
    """
    Improves a given tour (permutation) using the 2-Opt local search
    heuristic with a 'first improvement' strategy. 
    """
    n_cities = len(tour)
    best_tour = tour.copy()
    best_cost = calculate_tour_cost(best_tour, dist_matrix)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_cities - 2):
            for j in range(i + 2, n_cities):
                A, B = best_tour[i], best_tour[i+1]
                C, D = best_tour[j], best_tour[(j + 1) % n_cities]
                
                old_cost = dist_matrix[A, B] + dist_matrix[C, D]
                new_cost = dist_matrix[A, C] + dist_matrix[B, D]
                cost_change = new_cost - old_cost
                
                if cost_change < -1e-9:
                    new_tour = best_tour.copy()
                    new_tour[i+1 : j+1] = new_tour[j : i : -1] # Reversing slice
                    
                    best_tour = new_tour
                    best_cost += cost_change
                    improved = True
                    break
            if improved:
                break
                
    return best_tour, best_cost

# --- Section IV.D: Hybrid Particle Class ---

class Particle:
    """
    Represents a single particle in the Hybrid-PSO (HPSO).
    Integrates DPSO (global search) with 2-Opt (local search).
    [3, 21, 39, 41]
    """
    def __init__(self, n_cities, dist_matrix):
        self.dist_matrix = dist_matrix
        
        # 1. Initialize Position (a random permutation)
        self.position = np.random.permutation(n_cities)
        
        # 2. Apply 2-Opt to the initial random position
        self.position, self.cost = apply_2_opt(self.position, 
                                               self.dist_matrix)
        
        # 3. Initialize pBest (personal best)
        self.pbest_position = self.position.copy()
        self.pbest_cost = self.cost
        
        # 4. Initialize Velocity
        self.velocity = []

    def update(self, gbest_position, w, c1, c2):
        """
        Update the particle's velocity and position using the
        full HPSO (DPSO + 2-Opt) update cycle.
        """
        
        # 1. DPSO Global Search (Exploration)
        v_inertia = multiply_velocity(w, self.velocity)
        v_pbest = multiply_velocity(c1, 
            subtract_permutations(self.pbest_position, self.position))
        v_gbest = multiply_velocity(c2,
            subtract_permutations(gbest_position, self.position))
        
        # 2. Combine velocities
        self.velocity = add_velocities(v_inertia, 
                                     add_velocities(v_pbest, v_gbest))
        
        # 3. Apply velocity to get new position
        self.position = apply_velocity_to_position(self.position, self.velocity)
        
        # 4. 2-Opt Local Search (Exploitation)
        self.position, self.cost = apply_2_opt(self.position, 
                                               self.dist_matrix)
        
        # 5. Update pBest
        if self.cost < self.pbest_cost:
            self.pbest_cost = self.cost
            self.pbest_position = self.position.copy()

# --- Section V.A: Main Solver Class ---

class HybridPSOSolver:
    """
    The main HPSO-TSP Solver class.
    Orchestrates the swarm and manages the optimization process.
    """
    def __init__(self, n_particles, n_iterations, w=0.7, c1=1.5, c2=1.5):
        """
        Initialize the solver with configurable parameters. 
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w   # Inertia weight
        self.c1 = c1 # Cognitive (personal) coefficient
        self.c2 = c2 # Social (global) coefficient
        

        self.swarm = []
        self.gbest_position = None
        self.gbest_cost = np.inf
        
        # Store gBest cost at each iteration for plotting
        self.convergence_curve = []

    def solve(self, cities):
        """
        Run the HPSO-TSP optimization.
        """
        n_cities = len(cities)
        dist_matrix = calculate_distance_matrix(cities)
        
        # 1. Initialize the swarm of particles
        self.swarm = [Particle(n_cities, dist_matrix) for _ in range(self.n_particles)]
        
        # 2. Find the initial gBest
        # gBest is the best pBest from the initialized swarm
        for p in self.swarm:
            if p.pbest_cost < self.gbest_cost:
                self.gbest_cost = p.pbest_cost
                self.gbest_position = p.pbest_position.copy()
                
        print(f"Initial best cost (after 2-Opt): {self.gbest_cost:.2f}")
        
        # 3. Run the main optimization loop 
        for t in range(self.n_iterations):
            # Update all particles in the swarm
            for p in self.swarm:
                p.update(self.gbest_position, self.w, self.c1, self.c2)
            
            # Check for a new gBest after the update step
            for p in self.swarm:
                if p.pbest_cost < self.gbest_cost:
                    self.gbest_cost = p.pbest_cost
                    self.gbest_position = p.pbest_position.copy()
            
            # Record the best cost for this iteration
            self.convergence_curve.append(self.gbest_cost)
            
            if (t + 1) % 10 == 0:
                print(f"Iteration {t+1}/{self.n_iterations}, Best Cost: {self.gbest_cost:.2f}")
                
        print(f"Optimization finished. Final best cost: {self.gbest_cost:.2f}")
        return self.gbest_position, self.gbest_cost, self.convergence_curve
# --- Section V.B: Visualization Functions ---

def plot_tour(tour, cities, title=""):
    """
    Plots the final optimized tour.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the tour path
    # Re-order cities according to the tour
    tour_cities = cities[tour, :]
    
    # Add the starting city to the end to close the loop
    tour_cities = np.vstack([tour_cities, tour_cities[0, :]])
    
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-')
    
    # Plot the city nodes
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=50)
    
    # Annotate cities
    for i, city in enumerate(cities):
        plt.text(city[0] + 0.5, city[1] + 0.5, str(i))
        
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

def plot_convergence(curve):
    """
    Plots the convergence curve (gBest cost over iterations).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(curve)
    plt.title("HPSO-TSP Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Cost (Fitness)")
    plt.grid(True)
# --- Section V.C: Example Execution ---

if __name__ == "__main__":
    # 1. Setup Problem
    N_CITIES = 30   # Number of cities
    MAP_SIZE = 100
    cities = create_cities(N_CITIES, map_size=MAP_SIZE, seed=42)
    
    # 2. Setup Solver (Configurable Parameters )
    solver = HybridPSOSolver(
        n_particles=20,   # Population size
        n_iterations=200, # Number of generations
        w=0.729,          # Inertia weight (a common default)
        c1=1.494,         # Cognitive coefficient
        c2=1.494          # Social coefficient
    )
    
    # 3. Run Solver
    best_tour, best_cost, curve = solver.solve(cities)
    
    # 4. Report and Visualize Results 
    print("\n" + "="*30)
    print("      Final Results")
    print("="*30)
    print(f"Best tour (permutation): \n{best_tour}")
    print(f"Best cost: {best_cost:.4f}")
    
    # Plot the convergence curve
    plot_convergence(curve)
    
    # Plot the final best tour
    plot_tour(best_tour, cities, f"Final Tour (Cost: {best_cost:.4f})")
    
    plt.show()