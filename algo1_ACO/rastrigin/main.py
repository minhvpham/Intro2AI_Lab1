import numpy as np
import matplotlib.pyplot as plt
from visualization import (plot_convergence, plot_rastrigin_surface_2d, 
                          plot_convergence_with_solutions, plot_animated_convergence,
                          plot_animated_convergence_simple)
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.Continuous_functions import rastrigin

# --- 2 ---
class ACOR_Solver:
    def __init__(self, cost_function, n_dims, bounds,
                archive_size = 50,
                sample_size = 100,
                n_iterations = 250,
                q = 0.5,
                zeta = 1.0,
                track_archive = False):

        self.cost_function = cost_function
        self.n_dims = n_dims
        self.bounds_low, self.bounds_high = bounds
        self.k = archive_size
        self.m = sample_size
        self.n_iterations = n_iterations
        self.q = q      # Intensification factor (selection pressure)
        self.zeta = zeta  # Deviation-distance ratio
        self.track_archive = track_archive  # Whether to save archive history
        
        # The "Pheromone Model" is the solution archive
        # (k, n_dims + 1) -> [sol_vector, cost]
        self.archive = np.zeros((self.k, self.n_dims + 1))
        
        self.best_solution = None
        self.best_cost = np.inf
        self.convergence_history = []
        self.archive_history = []  # Store archive snapshots for visualization

    def _initialize_archive(self):
        """Fill the archive with k random solutions."""
        for i in range(self.k):
            solution = np.random.uniform(self.bounds_low, self.bounds_high, self.n_dims)
            cost = self.cost_function(solution)
            self.archive[i] = [*solution, cost]
        
        self._sort_archive()

    def _sort_archive(self):
        """Sort the archive by cost (last column), best-to-worst."""
        self.archive = self.archive[self.archive[:, -1].argsort()]
        
        if self.archive[0, -1] < self.best_cost:
            self.best_cost = self.archive[0, -1]
            self.best_solution = self.archive[0, :-1]

    def run(self):
        self._initialize_archive()
        
        # Store initial archive if tracking
        if self.track_archive:
            self.archive_history.append(self.archive.copy())
        
        for _ in range(self.n_iterations):
            # 1. Probabilistically select guides and generate new solutions
            new_solutions = self._generate_solutions()
            
            # 2. "Evaporation" / Archive Update 
            # Add new solutions, sort, and truncate the worst
            self.archive = np.vstack((self.archive, new_solutions))
            self._sort_archive()
            self.archive = self.archive[:self.k] # Keep only the best k
            
            self.convergence_history.append(self.best_cost)
            
            # Store archive snapshot if tracking
            if self.track_archive:
                self.archive_history.append(self.archive.copy())
            
        return self.best_solution, self.best_cost

    def _generate_solutions(self):
        """Generate 'm' new solutions by sampling around the archive."""
        new_solutions = np.zeros((self.m, self.n_dims + 1))
        
        # Calculate selection probabilities (weights) for archive guides
        # This is the "pheromone" part - better solutions are more likely
        weights = (1 / (self.q * self.k * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * (np.arange(self.k)**2) / (self.q**2 * self.k**2))
        weights /= np.sum(weights)

        for i in range(self.m):
            # 1. Probabilistically select a guide solution
            guide_index = np.random.choice(self.k, p=weights)
            guide_solution = self.archive[guide_index, :-1]
            
            # 2. Calculate standard deviations for sampling (Gaussian PDF)
            # This is the "search radius"
            std_devs = np.zeros(self.n_dims)
            for dim in range(self.n_dims):
                # Calculate avg distance to other archive solutions
                avg_dist = np.mean(np.abs(self.archive[:, dim] - guide_solution[dim]))
                std_devs[dim] = self.zeta * avg_dist
            
            # 3. Construct new solution by sampling 
            new_solution_vec = np.random.normal(loc=guide_solution, scale=std_devs)
            
            # Clip to bounds
            new_solution_vec = np.clip(new_solution_vec, self.bounds_low, self.bounds_high)
            
            # Evaluate cost
            cost = self.cost_function(new_solution_vec)
            new_solutions[i] = [*new_solution_vec, cost]
            
        return new_solutions

# --- Main execution block for ACOR ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACOR Algorithm for Rastrigin Function Optimization")
    print("="*70)
    
    # --- Example 1: High-dimensional optimization (5D) with basic visualization ---
    print("\n[Example 1] Solving 5-D Rastrigin function...")
    DIMS_5D = 5
    BOUNDS = [-5.12, 5.12]
    ARCHIVE_SIZE = 30
    SAMPLE_SIZE = 50
    ITERATIONS = 100
    
    acor_solver_5d = ACOR_Solver(
        cost_function=rastrigin,
        n_dims=DIMS_5D,
        bounds=BOUNDS,
        archive_size=ARCHIVE_SIZE,
        sample_size=SAMPLE_SIZE,
        n_iterations=ITERATIONS,
        track_archive=False  # Don't track for high-dimensional
    )
    
    best_sol_5d, best_cost_5d = acor_solver_5d.run()
    
    print("\n--- Results (5-D) ---")
    print(f"Best solution cost: {best_cost_5d:.6f}")
    print(f"Best solution vector: {best_sol_5d}")
    print(f"Distance from global optimum (0): {np.linalg.norm(best_sol_5d):.6f}")
    
    # Plot basic convergence
    print("\nGenerating convergence plot...")
    plot_convergence(acor_solver_5d.convergence_history)
    
    # --- Example 2: 2D optimization with enhanced visualization ---
    print("\n" + "-"*70)
    print("[Example 2] Solving 2-D Rastrigin function with enhanced visualization...")
    DIMS_2D = 2
    ITERATIONS_2D = 20
    
    # First, visualize the Rastrigin surface
    print("\nVisualizing the Rastrigin function surface...")
    plot_rastrigin_surface_2d(bounds=(BOUNDS[0], BOUNDS[1]))
    
    # Run optimization with archive tracking
    acor_solver_2d = ACOR_Solver(
        cost_function=rastrigin,
        n_dims=DIMS_2D,
        bounds=BOUNDS,
        archive_size=ARCHIVE_SIZE,
        sample_size=SAMPLE_SIZE,
        n_iterations=ITERATIONS_2D,
        q=0.5,
        zeta=1.0,
        track_archive=True  # Enable archive tracking for visualization
    )
    
    best_sol_2d, best_cost_2d = acor_solver_2d.run()
    
    print("\n--- Results (2-D) ---")
    print(f"Best solution cost: {best_cost_2d:.6f}")
    print(f"Best solution vector: {best_sol_2d}")
    print(f"Distance from global optimum (0, 0): {np.linalg.norm(best_sol_2d):.6f}")
    
    # Plot animated convergence with full details (2D)
    print("\nGenerating animated convergence with solution evolution (1 second per iteration)...")
    plot_animated_convergence(
        history=acor_solver_2d.convergence_history,
        archive_history=acor_solver_2d.archive_history,
        bounds=(BOUNDS[0], BOUNDS[1]),
        pause_time=1.0,
        cost_function_name="Rastrigin (2D)"
    )
    
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)