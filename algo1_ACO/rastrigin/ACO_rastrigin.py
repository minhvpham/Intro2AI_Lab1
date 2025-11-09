import numpy as np


# --- ACOR Solver ---
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

