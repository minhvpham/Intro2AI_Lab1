import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import random


# --- Firefly Algorithm (FA) Implementation ---
class FireflyAlgorithm:
    def __init__(self, objective_func, dimensions, lower_bound, upper_bound, 
                 n_fireflies=50, max_iterations=100, 
                 alpha=0.5, beta0=1.0, gamma=0.01):
        """
        Initialize the Firefly Algorithm.
        - objective_func: The function to be minimized.
        - dimensions: Number of dimensions of the problem (d).
        - lower_bound: Lower bound of the search space.
        - upper_bound: Upper bound of the search space.
        - n_fireflies: Population size (N).
        - max_iterations: Number of generations (T).
        - alpha: Randomization parameter (controls step size).
        - beta0: Attractiveness at r=0.
        - gamma: Light absorption coefficient.
        """
        self.objective_func = objective_func
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        # Initialize firefly positions (population)
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, 
                                           (self.n_fireflies, self.dimensions))
        
        # Calculate initial fitness for each firefly
        self.fitness = np.apply_along_axis(self.objective_func, 1, self.positions)
        
        # Find the initial best
        self.best_index = np.argmin(self.fitness)
        self.global_best_pos = self.positions[self.best_index].copy()
        self.global_best_fitness = self.fitness[self.best_index]
        
        # History for plotting
        self.convergence_history = []
        self.position_history = [] # For animation
        self.intensity_history = [] # Store fitness (intensity) values
        self.best_position_history = [] # Store global best position over time

    def run(self):
        """
        Run the optimization main loop.
        """
        start_time = time.time()
        
        for t in range(self.max_iterations):
            self.alpha *= 0.97
            # Store current state for animation
            self.position_history.append(self.positions.copy())
            self.intensity_history.append(self.fitness.copy())
            self.best_position_history.append(self.global_best_pos.copy())
            
            # O(N^2) all-to-all comparison
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    
                    # Move firefly i towards j ONLY if j is brighter (lower fitness)
                    if self.fitness[j] < self.fitness[i]:
                        # Calculate Euclidean distance
                        r_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                        
                        # Calculate attractiveness beta(r)
                        beta = self.beta0 * np.exp(-self.gamma * r_ij**2)
                        
                        # Generate random step (epsilon)
                        random_step = self.alpha * (np.random.rand(self.dimensions) - 0.5) * (self.upper_bound - self.lower_bound)
                        
                        # The movement equation
                        self.positions[i] += beta * (self.positions[j] - self.positions[i]) + random_step
            
            # After all moves, apply boundary constraints 
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            
            # Evaluate new positions
            self.fitness = np.apply_along_axis(self.objective_func, 1, self.positions)
            
            # Find the current best
            current_best_index = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_index]
            
            # Update global best
            if current_best_fitness < self.global_best_fitness:
                self.global_best_fitness = current_best_fitness
                self.global_best_pos = self.positions[current_best_index].copy()
            
            self.convergence_history.append(self.global_best_fitness)
            
            if t % 10 == 0:
                print(f"Iteration {t}: Best Fitness = {self.global_best_fitness}")
        
        end_time = time.time()
        print(f"Optimization finished in {end_time - start_time:.2f}s")
        print(f"Final Best Position: {self.global_best_pos}")
        print(f"Final Best Fitness: {self.global_best_fitness}")
        
        return self.global_best_pos, self.global_best_fitness, self.convergence_history
