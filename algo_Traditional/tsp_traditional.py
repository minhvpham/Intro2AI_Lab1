"""
Traditional Search Algorithms for Traveling Salesman Problem (TSP)
Implements Hill Climbing, A* Search, and Genetic Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import time


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_cities(n_cities, map_size=100, seed=42):
    """Generate random city coordinates."""
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2) * map_size
    return cities


def calculate_distance_matrix(cities):
    """Calculate pairwise distance matrix."""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def calculate_tour_cost(tour, dist_matrix):
    """Calculate total cost of a tour."""
    cost = 0
    for i in range(len(tour)):
        cost += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return cost


def plot_tour(cities, tour, title="TSP Tour", cost=None):
    """Visualize a TSP tour."""
    plt.figure(figsize=(10, 8))
    
    # Plot tour path
    tour_cities = cities[tour]
    tour_cities = np.vstack([tour_cities, tour_cities[0]])
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=1.5, alpha=0.7)
    
    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
    
    # Mark start city
    plt.scatter(cities[tour[0], 0], cities[tour[0], 1], 
                c='green', s=200, marker='*', zorder=6, label='Start')
    
    # Annotate cities
    for i, city in enumerate(cities):
        plt.text(city[0] + 1, city[1] + 1, str(i), fontsize=9)
    
    title_str = title
    if cost is not None:
        title_str += f" (Cost: {cost:.2f})"
    
    plt.title(title_str, fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 1. HILL CLIMBING FOR TSP
# ============================================================================

class HillClimbingTSP:
    """
    Hill Climbing for TSP using 2-opt local search.
    Uses random restarts to escape local minima.
    """
    
    def __init__(self, cities, n_restarts=10, max_iterations=1000):
        """
        Args:
            cities: Array of city coordinates (n_cities, 2)
            n_restarts: Number of random restarts
            max_iterations: Maximum iterations per restart
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.n_restarts = n_restarts
        self.max_iterations = max_iterations
        self.dist_matrix = calculate_distance_matrix(cities)
        
        self.best_tour = None
        self.best_cost = np.inf
        self.convergence_history = []
        
    def _two_opt_step(self, tour):
        """Perform one step of 2-opt improvement."""
        improved = False
        best_tour = tour.copy()
        best_cost = calculate_tour_cost(tour, self.dist_matrix)
        
        for i in range(self.n_cities - 1):
            for j in range(i + 2, self.n_cities):
                # Calculate cost change of reversing segment [i+1:j+1]
                new_tour = tour.copy()
                new_tour[i+1:j+1] = tour[j:i:-1]
                
                new_cost = calculate_tour_cost(new_tour, self.dist_matrix)
                
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True
                    return best_tour, best_cost, True  # First improvement
        
        return best_tour, best_cost, improved
    
    def _two_opt_full(self, tour):
        """Apply 2-opt until no improvement."""
        current_tour = tour.copy()
        current_cost = calculate_tour_cost(current_tour, self.dist_matrix)
        
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            current_tour, current_cost, improved = self._two_opt_step(current_tour)
            iterations += 1
        
        return current_tour, current_cost
    
    def run(self):
        """Execute hill climbing with random restarts."""
        print(f"\nRunning Hill Climbing TSP with {self.n_restarts} restarts...")
        
        for restart in range(self.n_restarts):
            # Random initial tour
            current_tour = np.random.permutation(self.n_cities)
            
            # Apply 2-opt
            current_tour, current_cost = self._two_opt_full(current_tour)
            
            # Update best
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_tour = current_tour
            
            self.convergence_history.append(self.best_cost)
            print(f"  Restart {restart + 1}/{self.n_restarts} - Local minimum: {current_cost:.2f}")
        
        print(f"Hill Climbing completed. Best cost: {self.best_cost:.2f}")
        return self.best_tour, self.best_cost


# ============================================================================
# 2. A* SEARCH FOR TSP
# ============================================================================

class AStarTSP:
    """
    A* Search for TSP using MST-based heuristic.
    Note: Only practical for small TSP instances (n <= 15 cities)
    """
    
    def __init__(self, cities, time_limit=60):
        """
        Args:
            cities: Array of city coordinates (n_cities, 2)
            time_limit: Maximum time in seconds
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.dist_matrix = calculate_distance_matrix(cities)
        self.time_limit = time_limit
        
        self.best_tour = None
        self.best_cost = np.inf
        self.nodes_explored = 0
        
    def _mst_heuristic(self, visited_set, current_city):
        """
        Compute MST-based lower bound for remaining cities.
        This is an admissible heuristic for A*.
        """
        unvisited = [i for i in range(self.n_cities) if i not in visited_set]
        
        if len(unvisited) == 0:
            # Return to start
            return self.dist_matrix[current_city, 0]
        
        # Prim's algorithm for MST
        if len(unvisited) == 1:
            return self.dist_matrix[current_city, unvisited[0]] + self.dist_matrix[unvisited[0], 0]
        
        # Simple MST lower bound
        total_mst = 0
        remaining = set(unvisited)
        
        # Add edge from current city to nearest unvisited
        min_to_unvisited = min(self.dist_matrix[current_city, i] for i in unvisited)
        total_mst += min_to_unvisited
        
        # MST of unvisited cities
        while remaining:
            if len(remaining) == len(unvisited):
                # Start from arbitrary city
                u = unvisited[0]
                remaining.remove(u)
                in_tree = {u}
            else:
                # Find minimum edge connecting tree to remaining
                min_edge = np.inf
                min_v = None
                for u in in_tree:
                    for v in remaining:
                        if self.dist_matrix[u, v] < min_edge:
                            min_edge = self.dist_matrix[u, v]
                            min_v = v
                
                total_mst += min_edge
                remaining.remove(min_v)
                in_tree.add(min_v)
        
        # Add edge back to start from nearest unvisited
        min_back = min(self.dist_matrix[i, 0] for i in unvisited)
        total_mst += min_back
        
        return total_mst
    
    def run(self):
        """
        Execute A* search for TSP.
        State: (cost_so_far, heuristic, current_city, visited_set, path)
        """
        if self.n_cities > 15:
            print(f"\nWarning: A* for TSP with {self.n_cities} cities is impractical!")
            print("A* is only suitable for small instances (n <= 15).")
            print("Returning a greedy solution instead...\n")
            return self._greedy_solution()
        
        print(f"\nRunning A* Search for TSP ({self.n_cities} cities)...")
        start_time = time.time()
        
        # Priority queue: (f_score, g_score, current_city, visited_set, path)
        start_city = 0
        initial_state = (
            0,  # f_score (will be updated)
            0,  # g_score
            start_city,
            frozenset([start_city]),
            [start_city]
        )
        
        # Calculate initial heuristic
        h = self._mst_heuristic(set([start_city]), start_city)
        initial_state = (h, 0, start_city, frozenset([start_city]), [start_city])
        
        pq = []
        heappush(pq, initial_state)
        
        visited_states = set()
        
        while pq and (time.time() - start_time) < self.time_limit:
            f_score, g_score, current_city, visited_set, path = heappop(pq)
            
            self.nodes_explored += 1
            
            # Goal test: all cities visited
            if len(visited_set) == self.n_cities:
                # Add return to start
                total_cost = g_score + self.dist_matrix[current_city, start_city]
                
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_tour = np.array(path)
                    print(f"  Found solution with cost: {total_cost:.2f} (nodes explored: {self.nodes_explored})")
                
                continue
            
            # State identifier
            state_id = (current_city, visited_set)
            if state_id in visited_states:
                continue
            visited_states.add(state_id)
            
            # Expand neighbors
            for next_city in range(self.n_cities):
                if next_city not in visited_set:
                    new_g = g_score + self.dist_matrix[current_city, next_city]
                    new_visited = visited_set | {next_city}
                    new_path = path + [next_city]
                    
                    # Calculate heuristic
                    h = self._mst_heuristic(set(new_visited), next_city)
                    new_f = new_g + h
                    
                    new_state = (new_f, new_g, next_city, frozenset(new_visited), new_path)
                    heappush(pq, new_state)
            
            # Progress update
            if self.nodes_explored % 10000 == 0:
                print(f"  Explored {self.nodes_explored} nodes... (queue size: {len(pq)})")
        
        elapsed = time.time() - start_time
        
        if self.best_tour is not None:
            print(f"A* Search completed in {elapsed:.2f}s")
            print(f"Nodes explored: {self.nodes_explored}")
            print(f"Best cost: {self.best_cost:.2f}")
        else:
            print(f"A* Search time limit reached ({elapsed:.2f}s)")
            print("No complete solution found - returning greedy solution")
            return self._greedy_solution()
        
        return self.best_tour, self.best_cost
    
    def _greedy_solution(self):
        """Greedy nearest neighbor as fallback."""
        tour = [0]
        unvisited = set(range(1, self.n_cities))
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.dist_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        cost = calculate_tour_cost(tour, self.dist_matrix)
        print(f"Greedy solution cost: {cost:.2f}")
        
        return np.array(tour), cost


# ============================================================================
# 3. GENETIC ALGORITHM FOR TSP
# ============================================================================

class GeneticAlgorithmTSP:
    """
    Genetic Algorithm for TSP using order-based encoding.
    """
    
    def __init__(self, cities, population_size=100, n_generations=200,
                 crossover_rate=0.9, mutation_rate=0.2, elite_size=5,
                 tournament_size=5):
        """
        Args:
            cities: Array of city coordinates (n_cities, 2)
            population_size: Number of tours in population
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best tours to preserve
            tournament_size: Tournament size for selection
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        self.dist_matrix = calculate_distance_matrix(cities)
        
        self.best_tour = None
        self.best_cost = np.inf
        self.convergence_history = []
        
        self.population = []
        self.fitness = []
        
    def _initialize_population(self):
        """Create initial random population."""
        self.population = [
            np.random.permutation(self.n_cities) 
            for _ in range(self.population_size)
        ]
        
    def _evaluate_fitness(self):
        """Evaluate fitness for entire population."""
        self.fitness = [
            calculate_tour_cost(tour, self.dist_matrix) 
            for tour in self.population
        ]
        
        # Update best
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_cost:
            self.best_cost = self.fitness[best_idx]
            self.best_tour = self.population[best_idx].copy()
        
    def _tournament_selection(self):
        """Select tour using tournament selection."""
        tournament_idx = np.random.choice(
            self.population_size, self.tournament_size, replace=False
        )
        tournament_fitness = [self.fitness[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _order_crossover(self, parent1, parent2):
        """
        Order Crossover (OX) for permutation encoding.
        Preserves relative order of cities from parents.
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = self.n_cities
        
        # Select two random cut points
        cx_point1, cx_point2 = sorted(np.random.choice(size, 2, replace=False))
        
        # Create children
        child1 = np.full(size, -1)
        child2 = np.full(size, -1)
        
        # Copy segment from parents
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        # Fill remaining positions
        def fill_child(child, parent):
            pos = cx_point2
            for city in np.concatenate([parent[cx_point2:], parent[:cx_point2]]):
                if city not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = city
                    pos += 1
            return child
        
        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)
        
        return child1.astype(int), child2.astype(int)
    
    def _swap_mutation(self, tour):
        """Swap mutation: randomly swap two cities."""
        mutated = tour.copy()
        
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(self.n_cities, 2, replace=False)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def _inversion_mutation(self, tour):
        """Inversion mutation: reverse a random segment."""
        mutated = tour.copy()
        
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = sorted(np.random.choice(self.n_cities, 2, replace=False))
            mutated[idx1:idx2+1] = mutated[idx1:idx2+1][::-1]
        
        return mutated
    
    def run(self):
        """Execute genetic algorithm."""
        print(f"\nRunning Genetic Algorithm for TSP...")
        print(f"  Population: {self.population_size}, Generations: {self.n_generations}")
        
        # Initialize
        self._initialize_population()
        self._evaluate_fitness()
        self.convergence_history.append(self.best_cost)
        
        for generation in range(self.n_generations):
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(self.fitness)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._order_crossover(parent1, parent2)
                
                # Mutation (alternating between swap and inversion)
                if len(new_population) % 2 == 0:
                    child1 = self._swap_mutation(child1)
                    child2 = self._swap_mutation(child2)
                else:
                    child1 = self._inversion_mutation(child1)
                    child2 = self._inversion_mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace population
            self.population = new_population[:self.population_size]
            
            # Evaluate
            self._evaluate_fitness()
            self.convergence_history.append(self.best_cost)
            
            if (generation + 1) % 50 == 0:
                print(f"  Generation {generation + 1}/{self.n_generations} - Best cost: {self.best_cost:.2f}")
        
        print(f"Genetic Algorithm completed. Best cost: {self.best_cost:.2f}")
        return self.best_tour, self.best_cost


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_convergence_comparison(histories, labels, title="TSP Algorithm Comparison"):
    """Plot convergence curves."""
    plt.figure(figsize=(12, 6))
    
    for history, label in zip(histories, labels):
        if len(history) > 0:
            plt.plot(history, label=label, linewidth=2, marker='o', markersize=4, markevery=max(1, len(history)//20))
    
    plt.xlabel('Iteration/Generation/Restart', fontsize=12)
    plt.ylabel('Best Tour Cost', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRADITIONAL ALGORITHMS FOR TSP")
    print("="*70)
    
    # Problem setup
    N_CITIES = 20  # Use smaller number for A* (<=15)
    MAP_SIZE = 100
    
    print(f"\nGenerating TSP instance with {N_CITIES} cities...")
    cities = create_cities(N_CITIES, map_size=MAP_SIZE, seed=42)
    
    convergence_histories = []
    labels = []
    
    # --- 1. Hill Climbing ---
    hc_tsp = HillClimbingTSP(cities, n_restarts=10, max_iterations=1000)
    hc_tour, hc_cost = hc_tsp.run()
    convergence_histories.append(hc_tsp.convergence_history)
    labels.append('Hill Climbing')
    
    print(f"\n[Hill Climbing Results]")
    print(f"  Best tour cost: {hc_cost:.2f}")
    print(f"  Best tour: {hc_tour}")
    
    # Visualize Hill Climbing solution
    plot_tour(cities, hc_tour, "Hill Climbing TSP Solution", hc_cost)
    
    # --- 2. A* Search (only for small instances) ---
    if N_CITIES <= 15:
        astar_tsp = AStarTSP(cities, time_limit=60)
        astar_tour, astar_cost = astar_tsp.run()
        
        print(f"\n[A* Search Results]")
        print(f"  Best tour cost: {astar_cost:.2f}")
        print(f"  Best tour: {astar_tour}")
        print(f"  Nodes explored: {astar_tsp.nodes_explored}")
        
        # Visualize A* solution
        plot_tour(cities, astar_tour, "A* Search TSP Solution", astar_cost)
    else:
        print(f"\n[A* Search] Skipped (N_CITIES={N_CITIES} > 15)")
        print("  A* is only practical for small TSP instances")
    
    # --- 3. Genetic Algorithm ---
    ga_tsp = GeneticAlgorithmTSP(
        cities,
        population_size=100,
        n_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_size=5,
        tournament_size=5
    )
    ga_tour, ga_cost = ga_tsp.run()
    convergence_histories.append(ga_tsp.convergence_history)
    labels.append('Genetic Algorithm')
    
    print(f"\n[Genetic Algorithm Results]")
    print(f"  Best tour cost: {ga_cost:.2f}")
    print(f"  Best tour: {ga_tour}")
    
    # Visualize GA solution
    plot_tour(cities, ga_tour, "Genetic Algorithm TSP Solution", ga_cost)
    
    # --- Comparison Plot ---
    print("\nGenerating convergence comparison...")
    plot_convergence_comparison(
        convergence_histories, labels,
        f"TSP ({N_CITIES} cities) - Algorithm Comparison"
    )
    
    # --- Summary Table ---
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Algorithm':<25} {'Best Cost':<15} {'Notes'}")
    print("-"*70)
    print(f"{'Hill Climbing (2-opt)':<25} {hc_cost:<15.2f} {len(hc_tsp.convergence_history)} restarts")
    if N_CITIES <= 15:
        print(f"{'A* Search':<25} {astar_cost:<15.2f} {astar_tsp.nodes_explored} nodes explored")
    print(f"{'Genetic Algorithm':<25} {ga_cost:<15.2f} {len(ga_tsp.convergence_history)} generations")
    print("="*70)
