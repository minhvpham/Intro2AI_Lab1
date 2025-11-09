"""
Traditional Search Algorithms for Continuous Optimization Problems
Implements Hill Climbing (Steepest Ascent) and Genetic Algorithm for functions like Rastrigin
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Benchmark Functions ---
def rastrigin(X):
    """Rastrigin function (minimization). Global minimum is 0.0 at X = [0, 0,...]."""
    A = 10
    n_dims = len(X)
    return A * n_dims + np.sum(X**2 - A * np.cos(2 * np.pi * X))


def sphere(X):
    """Sphere function (minimization). Global minimum is 0.0 at X = [0, 0,...]."""
    return np.sum(X**2)


# ============================================================================
# 1. HILL CLIMBING (STEEPEST ASCENT/DESCENT)
# ============================================================================

class HillClimbing:
    """
    Hill Climbing algorithm for continuous optimization.
    Uses steepest descent with random restarts to escape local minima.
    """
    
    def __init__(self, cost_function, n_dims, bounds, 
                 step_size=0.1, max_iterations=1000, 
                 n_restarts=10, adaptive_step=True):
        """
        Args:
            cost_function: The objective function to minimize
            n_dims: Number of dimensions
            bounds: [lower_bound, upper_bound] for search space
            step_size: Initial step size for neighborhood exploration
            max_iterations: Maximum iterations per restart
            n_restarts: Number of random restarts
            adaptive_step: Whether to reduce step size when stuck
        """
        self.cost_function = cost_function
        self.n_dims = n_dims
        self.bounds_low, self.bounds_high = bounds
        self.initial_step_size = step_size
        self.max_iterations = max_iterations
        self.n_restarts = n_restarts
        self.adaptive_step = adaptive_step
        
        self.best_solution = None
        self.best_cost = np.inf
        self.convergence_history = []
        
    def run(self):
        """Execute hill climbing with random restarts."""
        print(f"\nRunning Hill Climbing with {self.n_restarts} restarts...")
        
        for restart in range(self.n_restarts):
            # Random initialization
            current_solution = np.random.uniform(
                self.bounds_low, self.bounds_high, self.n_dims
            )
            current_cost = self.cost_function(current_solution)
            
            step_size = self.initial_step_size
            no_improvement_count = 0
            
            for iteration in range(self.max_iterations):
                # Generate neighbors by perturbing each dimension
                improved = False
                best_neighbor = current_solution.copy()
                best_neighbor_cost = current_cost
                
                # Explore neighborhood in all dimensions
                for dim in range(self.n_dims):
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        neighbor = current_solution.copy()
                        neighbor[dim] += direction * step_size
                        
                        # Clip to bounds
                        neighbor = np.clip(neighbor, self.bounds_low, self.bounds_high)
                        
                        neighbor_cost = self.cost_function(neighbor)
                        
                        # Keep track of best neighbor
                        if neighbor_cost < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = neighbor_cost
                            improved = True
                
                # Move to best neighbor if improvement found
                if improved:
                    current_solution = best_neighbor
                    current_cost = best_neighbor_cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                    # Adaptive step size reduction
                    if self.adaptive_step and no_improvement_count > 5:
                        step_size *= 0.5
                        no_improvement_count = 0
                        
                        # If step size too small, break
                        if step_size < 1e-6:
                            break
                
                # Update global best
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_solution = current_solution.copy()
                
                self.convergence_history.append(self.best_cost)
            
            print(f"  Restart {restart + 1}/{self.n_restarts} - Local minimum: {current_cost:.6f}")
        
        print(f"Hill Climbing completed. Best cost: {self.best_cost:.6f}")
        return self.best_solution, self.best_cost


# ============================================================================
# 2. GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for continuous optimization using real-valued encoding.
    """
    
    def __init__(self, cost_function, n_dims, bounds,
                 population_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.1,
                 elite_size=2, tournament_size=3):
        """
        Args:
            cost_function: The objective function to minimize
            n_dims: Number of dimensions
            bounds: [lower_bound, upper_bound] for search space
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
        """
        self.cost_function = cost_function
        self.n_dims = n_dims
        self.bounds_low, self.bounds_high = bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        self.best_solution = None
        self.best_cost = np.inf
        self.convergence_history = []
        
        # Initialize population
        self.population = None
        self.fitness = None
        
    def _initialize_population(self):
        """Create initial random population."""
        self.population = np.random.uniform(
            self.bounds_low, self.bounds_high,
            (self.population_size, self.n_dims)
        )
        
    def _evaluate_fitness(self):
        """Evaluate fitness for entire population."""
        self.fitness = np.array([
            self.cost_function(individual) 
            for individual in self.population
        ])
        
        # Update best solution
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_cost:
            self.best_cost = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
    
    def _tournament_selection(self):
        """Select individual using tournament selection."""
        # Randomly select tournament_size individuals
        tournament_indices = np.random.choice(
            self.population_size, self.tournament_size, replace=False
        )
        tournament_fitness = self.fitness[tournament_indices]
        
        # Return the best individual from tournament
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        """Simulated Binary Crossover (SBX) for real-valued encoding."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros(self.n_dims)
        child2 = np.zeros(self.n_dims)
        
        # BLX-alpha crossover (Blend Crossover)
        alpha = 0.5
        for i in range(self.n_dims):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            
            # Blend with alpha extension
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            child1[i] = np.random.uniform(lower, upper)
            child2[i] = np.random.uniform(lower, upper)
        
        # Clip to bounds
        child1 = np.clip(child1, self.bounds_low, self.bounds_high)
        child2 = np.clip(child2, self.bounds_low, self.bounds_high)
        
        return child1, child2
    
    def _mutate(self, individual):
        """Gaussian mutation for real-valued encoding."""
        mutated = individual.copy()
        
        for i in range(self.n_dims):
            if np.random.rand() < self.mutation_rate:
                # Gaussian mutation with adaptive sigma
                sigma = (self.bounds_high - self.bounds_low) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                
                # Clip to bounds
                mutated[i] = np.clip(mutated[i], self.bounds_low, self.bounds_high)
        
        return mutated
    
    def run(self):
        """Execute the genetic algorithm."""
        print(f"\nRunning Genetic Algorithm for {self.n_generations} generations...")
        
        # Initialize
        self._initialize_population()
        self._evaluate_fitness()
        self.convergence_history.append(self.best_cost)
        
        for generation in range(self.n_generations):
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(self.fitness)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace population
            self.population = np.array(new_population[:self.population_size])
            
            # Evaluate new population
            self._evaluate_fitness()
            self.convergence_history.append(self.best_cost)
            
            if (generation + 1) % 20 == 0:
                print(f"  Generation {generation + 1}/{self.n_generations} - Best fitness: {self.best_cost:.6f}")
        
        print(f"Genetic Algorithm completed. Best cost: {self.best_cost:.6f}")
        return self.best_solution, self.best_cost


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_convergence_comparison(convergence_histories, labels, title="Algorithm Comparison"):
    """Plot convergence curves for multiple algorithms."""
    plt.figure(figsize=(10, 6))
    
    for history, label in zip(convergence_histories, labels):
        plt.plot(history, label=label, linewidth=2)
    
    plt.xlabel('Iteration/Generation', fontsize=12)
    plt.ylabel('Best Cost (Fitness)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRADITIONAL ALGORITHMS FOR CONTINUOUS OPTIMIZATION")
    print("="*70)
    
    # Problem setup
    N_DIMS = 5
    BOUNDS = [-5.12, 5.12]
    COST_FUNCTION = rastrigin
    FUNCTION_NAME = "Rastrigin"
    
    print(f"\nProblem: {FUNCTION_NAME} function ({N_DIMS}D)")
    print(f"Search space: [{BOUNDS[0]}, {BOUNDS[1]}]^{N_DIMS}")
    print(f"Global minimum: 0.0 at origin")
    
    # --- 1. Hill Climbing ---
    hc = HillClimbing(
        cost_function=COST_FUNCTION,
        n_dims=N_DIMS,
        bounds=BOUNDS,
        step_size=0.5,
        max_iterations=200,
        n_restarts=10,
        adaptive_step=True
    )
    hc_solution, hc_cost = hc.run()
    
    print(f"\n[Hill Climbing Results]")
    print(f"  Best cost: {hc_cost:.6f}")
    print(f"  Best solution: {hc_solution}")
    print(f"  Distance from origin: {np.linalg.norm(hc_solution):.6f}")
    
    # --- 2. Genetic Algorithm ---
    ga = GeneticAlgorithm(
        cost_function=COST_FUNCTION,
        n_dims=N_DIMS,
        bounds=BOUNDS,
        population_size=50,
        n_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=2,
        tournament_size=3
    )
    ga_solution, ga_cost = ga.run()
    
    print(f"\n[Genetic Algorithm Results]")
    print(f"  Best cost: {ga_cost:.6f}")
    print(f"  Best solution: {ga_solution}")
    print(f"  Distance from origin: {np.linalg.norm(ga_solution):.6f}")
    
    # --- Visualization ---
    print("\nGenerating convergence comparison plot...")
    plot_convergence_comparison(
        convergence_histories=[hc.convergence_history, ga.convergence_history],
        labels=['Hill Climbing', 'Genetic Algorithm'],
        title=f'{FUNCTION_NAME} Function ({N_DIMS}D) - Algorithm Comparison'
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
