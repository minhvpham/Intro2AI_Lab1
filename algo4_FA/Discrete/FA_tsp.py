import numpy as np
import matplotlib.pyplot as plt



def calculate_distance_matrix(cities):
    """
    Calculates the N x N pairwise Euclidean distance matrix
    from a (N, D) city coordinate matrix, using only NumPy. [33]
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 0)
    return np.sqrt(dist_sq)

def calculate_tour_distance(tour, dist_matrix):
    """Calculates the total distance of a tour (permutation)."""
    distance = 0
    for i in range(len(tour)):
        # Add distance from current city to next city
        # Use % to wrap around to the start city
        distance += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return distance

def two_opt(tour, dist_matrix):
    """Apply 2-opt local search to improve a tour."""
    improved = True
    best_tour = tour.copy()
    best_distance = calculate_tour_distance(best_tour, dist_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                # Create new tour by reversing segment between i and j
                new_tour = best_tour.copy()
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                new_distance = calculate_tour_distance(new_tour, dist_matrix)
                
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return best_tour
# --- Discrete Firefly Algorithm (DFA) for TSP ---
def hamming_distance(tour1, tour2):
    """Calculates the Hamming distance between two permutations."""
    return np.sum(tour1!= tour2)

class DiscreteFireflyAlgorithm:
    def __init__(self, cities, n_fireflies=50, max_iterations=100, 
                 alpha=0.4, beta0=1.0, gamma=0.01, use_2opt=True):
        
        self.dist_matrix = calculate_distance_matrix(cities)
        self.cities = cities
        self.n_cities = len(cities)
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha  # Reduced default from 1.0 to 0.2
        self.beta0 = beta0
        self.gamma = gamma
        self.use_2opt = use_2opt
        
        # Initialize positions (tours) with 2-opt improvement
        self.tours = np.array([np.random.permutation(self.n_cities) for _ in range(self.n_fireflies)])
        
        # Apply 2-opt to initial tours if enabled
        if self.use_2opt:
            for i in range(self.n_fireflies):
                self.tours[i] = two_opt(self.tours[i], self.dist_matrix)
        
        # Calculate fitness (brightness)
        self.distances = np.array([calculate_tour_distance(t, self.dist_matrix) for t in self.tours])
        self.fitness = 1.0 / self.distances
        
        self.global_best_index = np.argmax(self.fitness)
        self.global_best_tour = self.tours[self.global_best_index].copy()
        self.global_best_distance = self.distances[self.global_best_index]
        
        self.convergence_history = []
        
    def _move_firefly(self, tour_i, tour_j, distance_ij):
        """Move tour_i 'towards' tour_j using guided swaps."""
        new_tour = tour_i.copy()
        
        # Calculate beta and number of swaps
        beta = self.beta0 * np.exp(-self.gamma * distance_ij**2)
        n_swaps = max(1, int(np.floor(beta * self.n_cities)))  # Reduced swap count
        
        # Find indices that differ
        diff_indices = np.where(new_tour != tour_j)[0]
        if len(diff_indices) == 0:
            return new_tour

        # Limit swaps to available differences
        n_swaps = min(n_swaps, len(diff_indices))

        for _ in range(n_swaps):
            # Pick a random differing index
            idx1 = np.random.choice(diff_indices)
            val_to_find = tour_j[idx1]
            
            # Find where this value is in new_tour
            idx2_result = np.where(new_tour == val_to_find)[0]
            if len(idx2_result) == 0:
                continue # Should not happen in a valid permutation
            idx2 = idx2_result[0]

            # Perform the swap
            new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
            
            # Update diff_indices
            diff_indices = np.where(new_tour != tour_j)[0]
            if len(diff_indices) == 0:
                break
                
        return new_tour

    def _random_move(self, tour):
        """Random move (swap mutation) for exploration."""
        if np.random.rand() < self.alpha:
            idx1, idx2 = np.random.choice(self.n_cities, 2, replace=False)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
        return tour

    def run(self):
        for t in range(self.max_iterations):
            new_tours = self.tours.copy()
            
            for i in range(self.n_fireflies):
                # Apply random move (exploration) with reduced probability
                if np.random.rand() < self.alpha:
                    new_tours[i] = self._random_move(new_tours[i])
                
                for j in range(self.n_fireflies):
                    # Move i towards brighter j
                    if self.fitness[j] > self.fitness[i]:
                        r_ij = hamming_distance(self.tours[i], self.tours[j])
                        moved_tour = self._move_firefly(new_tours[i], self.tours[j], r_ij)
                        
                        # Greedy selection: only accept if the move is good
                        moved_dist = calculate_tour_distance(moved_tour, self.dist_matrix)
                        if moved_dist < self.distances[i]:  # Direct comparison is better
                            new_tours[i] = moved_tour
                            self.distances[i] = moved_dist
                            self.fitness[i] = 1.0 / moved_dist
            
            # Apply 2-opt to some fireflies periodically
            if self.use_2opt and t % 10 == 0:
                # Apply 2-opt to worst performing fireflies
                worst_indices = np.argsort(self.fitness)[:self.n_fireflies // 4]
                for idx in worst_indices:
                    new_tours[idx] = two_opt(new_tours[idx], self.dist_matrix)
            
            self.tours = new_tours
            self.distances = np.array([calculate_tour_distance(t, self.dist_matrix) for t in self.tours])
            self.fitness = 1.0 / self.distances
            
            # Update global best
            current_best_idx = np.argmin(self.distances)
            if self.distances[current_best_idx] < self.global_best_distance:
                self.global_best_distance = self.distances[current_best_idx]
                self.global_best_tour = self.tours[current_best_idx].copy()
                
            self.convergence_history.append(self.global_best_distance)
            
            if t % 10 == 0:
                print(f"Iteration {t}: Best Distance = {self.global_best_distance:.2f}")
        
        # Final 2-opt on the best solution
        if self.use_2opt:
            self.global_best_tour = two_opt(self.global_best_tour, self.dist_matrix)
            self.global_best_distance = calculate_tour_distance(self.global_best_tour, self.dist_matrix)
        
        return self.global_best_tour, self.global_best_distance, self.convergence_history