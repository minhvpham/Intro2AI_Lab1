import numpy as np
import matplotlib.pyplot as plt

# --- ACO_TSP_Solver Class ---
class ACO_TSP_Solver:
    def __init__(self, coordinates, n_ants, n_iterations,
                 alpha=1.0, beta=2.0, rho=0.5, Q=100):
        
        self.coordinates = coordinates
        self.n_cities = len(coordinates)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        self.distances = self._calculate_distances()
        self.eta = 1.0 / (self.distances + 1e-10)
        
        # Initialize pheromones
        avg_dist = np.mean(self.distances[self.distances > 0])
        initial_tau = 1.0 / (self.n_cities * avg_dist)
        self.pheromones = np.full((self.n_cities, self.n_cities), initial_tau)
        
        self.best_path = None
        self.best_path_length = np.inf
        
        # To store convergence data
        self.convergence_history = []
        self.iteration_best_paths = []
        self.iteration_best_lengths = []

    def _calculate_distances(self):
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                # Use np.linalg.norm for Euclidean distance
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def run(self):
        """Main execution loop."""
        for iteration in range(self.n_iterations):
            all_paths, all_lengths = self._build_solutions()
            
            # Evaporation
            self.pheromones *= (1.0 - self.rho)
            
            # Deposition
            self._pheromone_deposition(all_paths, all_lengths)
            
            # Update best-so-far
            current_best_length = np.min(all_lengths)
            current_best_path = all_paths[np.argmin(all_lengths)]
            
            # Store iteration best
            self.iteration_best_paths.append(current_best_path)
            self.iteration_best_lengths.append(current_best_length)
            
            if current_best_length < self.best_path_length:
                self.best_path_length = current_best_length
                self.best_path = current_best_path
            
            self.convergence_history.append(self.best_path_length)
            print(f"Iteration {iteration + 1}/{self.n_iterations} - Best length: {self.best_path_length:.2f}")
                
        return self.best_path, self.best_path_length

    def _build_solutions(self):
        all_paths = []
        all_lengths = []
        for _ in range(self.n_ants):
            path, length = self._build_tour()
            all_paths.append(path)
            all_lengths.append(length)
        return all_paths, all_lengths

    def _build_tour(self, track_history=False):
        """Build a tour with optional step-by-step history tracking."""
        path = []
        visited = np.zeros(self.n_cities, dtype=bool)
        paths_history = []
        
        current_city = np.random.randint(0, self.n_cities)
        path.append(current_city)
        visited[current_city] = True
        if track_history:
            paths_history.append(path.copy())
        
        while len(path) < self.n_cities:
            next_city = self._select_next_city(current_city, visited)
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city
            if track_history:
                paths_history.append(path.copy())
            
        path.append(path[0]) # Return to start
        length = self._calculate_path_length(path)
        
        if track_history:
            return path, length, paths_history
        return path, length

    def _select_next_city(self, current_city, visited):
        """Probabilistically select the next city."""
        tau_values = self.pheromones[current_city, :]
        eta_values = self.eta[current_city, :]
        
        tau_pow = np.power(tau_values, self.alpha)
        eta_pow = np.power(eta_values, self.beta)
        
        probabilities = tau_pow * eta_pow
        probabilities[visited] = 0 # Mask visited cities
        
        sum_probs = np.sum(probabilities)
        
        if sum_probs == 0:
            # Fallback if all probabilities are zero
            unvisited_indices = np.where(visited == False)
            next_city = np.random.choice(unvisited_indices)
        else:
            probabilities /= sum_probs
            next_city = np.random.choice(self.n_cities, p=probabilities)
            
        return next_city

    def _calculate_path_length(self, path):
        length = 0
        for i in range(self.n_cities):
            length += self.distances[path[i], path[i+1]]
        return length

    def _pheromone_deposition(self, all_paths, all_lengths):
        """Update pheromones based on ant solutions."""
        for path, length in zip(all_paths, all_lengths):
            deposit_amount = self.Q / length
            for i in range(self.n_cities):
                city_a = path[i]
                city_b = path[i+1]
                self.pheromones[city_a, city_b] += deposit_amount
                self.pheromones[city_b, city_a] += deposit_amount # Symmetric

    def build_tour_with_tracking(self):
        """Build a tour and track ant movement step-by-step."""
        path = []
        visited = np.zeros(self.n_cities, dtype=bool)
        movement_steps = []
        
        current_city = np.random.randint(0, self.n_cities)
        path.append(current_city)
        visited[current_city] = True
        movement_steps.append((current_city, current_city))  # Start position
        
        while len(path) < self.n_cities:
            next_city = self._select_next_city(current_city, visited)
            path.append(next_city)
            visited[next_city] = True
            movement_steps.append((current_city, next_city))  # Movement edge
            current_city = next_city
        
        path.append(path[0])  # Return to start
        movement_steps.append((current_city, path[0]))
        length = self._calculate_path_length(path)
        
        return path, length, movement_steps
