"""
COMPREHENSIVE COMPARISON OF SWARM INTELLIGENCE VS TRADITIONAL ALGORITHMS

This script compares:
- Swarm Intelligence: ACO (ACOR), PSO, Firefly Algorithm (FA)
- Traditional: Hill Climbing, Genetic Algorithm

For both:
- Continuous Optimization (Rastrigin Function)
- Discrete Optimization (TSP)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from Continuous_functions import rastrigin
from tsp import create_cities

# Add parent directory to path so we can import from project root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import swarm algorithms
try:
    from algo1_ACO.rastrigin.ACO_rastrigin import ACOR_Solver
except Exception as e:
    print(f"Warning: Could not import ACOR_Solver - {e}")
    ACOR_Solver = None
    
try:
    from algo5_CS.continuous.main import cuckoo_search as CS_Solver
except Exception as e:
    print(f"Warning: Could not import CS_Solver - {e}")
    CS_Solver = None

try:
    from algo5_CS.discrete.main import cuckoo_search_tsp as CS_TSP_Solver
except Exception as e:
    print(f"Warning: Could not import CS_TSP_Solver - {e}")
    CS_TSP_Solver = None

try:
    from algo1_ACO.tsp.ACO import ACO_TSP_Solver
except:
    print("Warning: Could not import ACO_TSP_Solver")
    ACO_TSP_Solver = None

try:
    from algo2_PSO.continuous.pso import PSO
except:
    print("Warning: Could not import PSO")
    PSO = None

try:
    from algo2_PSO.discrete.pso_tsp import HybridPSOSolver
except:
    print("Warning: Could not import HybridPSOSolver")
    HybridPSOSolver = None

try:
    from algo4_FA.Continuous.FA import FireflyAlgorithm
except:
    print("Warning: Could not import FireflyAlgorithm")
    FireflyAlgorithm = None

try:
    from algo4_FA.Discrete.FA_tsp import DiscreteFireflyAlgorithm
except:
    print("Warning: Could not import DiscreteFireflyAlgorithm")
    DiscreteFireflyAlgorithm = None

try:
    from algo3_ABC.continuous.main import artificial_bee_colony
except:
    print("Warning: Could not import ABC continuous")
    artificial_bee_colony = None

try:
    from algo3_ABC.discrete.main import artificial_bee_colony_tsp
except:
    print("Warning: Could not import ABC discrete")
    artificial_bee_colony_tsp = None

# Import traditional algorithms
from algo_Traditional.continuous_traditional import HillClimbing, GeneticAlgorithm
from algo_Traditional.tsp_traditional import HillClimbingTSP, AStarTSP, GeneticAlgorithmTSP


# ============================================================================
# COMPARISON FRAMEWORK
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0  # psutil not available


class AlgorithmResults:
    """Store results from an algorithm run."""
    def __init__(self, name, solution, cost, time_elapsed, convergence_history=None, extra_info=None):
        self.name = name
        self.solution = solution
        self.cost = cost
        self.time_elapsed = time_elapsed
        self.convergence_history = convergence_history if convergence_history else []
        self.extra_info = extra_info if extra_info else {}
        
        # New metrics
        self.multiple_runs = []  # Store results from multiple runs for robustness
        self.memory_usage = 0    # Memory usage in MB
        self.convergence_speed = 0  # Iterations to reach threshold
        
    def add_run(self, cost, time_elapsed, convergence_history=None):
        """Add results from a single run for robustness analysis."""
        self.multiple_runs.append({
            'cost': cost,
            'time': time_elapsed,
            'convergence': convergence_history if convergence_history else []
        })
    
    def compute_robustness_metrics(self):
        """Compute robustness statistics from multiple runs."""
        if not self.multiple_runs:
            return None
        
        costs = [run['cost'] for run in self.multiple_runs]
        times = [run['time'] for run in self.multiple_runs]
        
        return {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'best_cost': np.min(costs),
            'worst_cost': np.max(costs),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'coefficient_of_variation': np.std(costs) / np.mean(costs) if np.mean(costs) != 0 else 0,
            'success_rate': sum(1 for c in costs if c < np.median(costs) * 1.1) / len(costs)
        }
    
    def compute_convergence_speed(self, threshold_percentile=0.9):
        """
        Compute convergence speed: iterations needed to reach threshold of final cost.
        Lower is faster.
        """
        if not self.convergence_history or len(self.convergence_history) < 2:
            self.convergence_speed = len(self.convergence_history) if self.convergence_history else 0
            return self.convergence_speed
        
        final_cost = self.convergence_history[-1]
        threshold = final_cost / threshold_percentile
        
        for i, cost in enumerate(self.convergence_history):
            if cost <= threshold:
                self.convergence_speed = i + 1
                return self.convergence_speed
        
        self.convergence_speed = len(self.convergence_history)
        return self.convergence_speed


def run_continuous_comparison(n_dims=5, bounds=(-5.12, 5.12), 
                               function=rastrigin, function_name="Rastrigin",
                               n_trials=1):
    """
    Compare all algorithms on continuous optimization problem.
    
    Args:
        n_dims: Problem dimensionality
        bounds: Search space bounds
        function: Objective function
        function_name: Name of the function
        n_trials: Number of independent runs for robustness testing
    """
    print("\n" + "="*80)
    print(f"CONTINUOUS OPTIMIZATION COMPARISON: {function_name} Function ({n_dims}D)")
    if n_trials > 1:
        print(f"Running {n_trials} independent trials for robustness analysis")
    print("="*80)
    
    results = []
    
    # --- 1. Hill Climbing ---
    print("\n[1/4] Running Hill Climbing...")
    hc_result = AlgorithmResults(
        name="Hill Climbing",
        solution=None,
        cost=float('inf'),
        time_elapsed=0,
        convergence_history=[],
        extra_info={'restarts': 10}
    )
    
    for trial in range(n_trials):
        if n_trials > 1:
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
        start_time = time.time()
        hc = HillClimbing(
            cost_function=function,
            n_dims=n_dims,
            bounds=bounds,
            step_size=0.5,
            max_iterations=200,
            n_restarts=10,
            adaptive_step=True
        )
        hc_solution, hc_cost = hc.run()
        hc_time = time.time() - start_time
        
        hc_result.add_run(hc_cost, hc_time, hc.convergence_history)
        
        # Keep best run as primary result
        if hc_cost < hc_result.cost:
            hc_result.solution = hc_solution
            hc_result.cost = hc_cost
            hc_result.time_elapsed = hc_time
            hc_result.convergence_history = hc.convergence_history
        
        if n_trials > 1:
            print(f"Cost: {hc_cost:.6f}")
    
    hc_result.compute_convergence_speed()
    results.append(hc_result)
    
    # --- 2. Genetic Algorithm ---
    print("\n[2/4] Running Genetic Algorithm...")
    ga_result = AlgorithmResults(
        name="Genetic Algorithm",
        solution=None,
        cost=float('inf'),
        time_elapsed=0,
        convergence_history=[],
        extra_info={'population': 50, 'generations': 100}
    )
    
    for trial in range(n_trials):
        if n_trials > 1:
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
        start_time = time.time()
        ga = GeneticAlgorithm(
            cost_function=function,
            n_dims=n_dims,
            bounds=bounds,
            population_size=50,
            n_generations=100,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        ga_solution, ga_cost = ga.run()
        ga_time = time.time() - start_time
        
        ga_result.add_run(ga_cost, ga_time, ga.convergence_history)
        
        if ga_cost < ga_result.cost:
            ga_result.solution = ga_solution
            ga_result.cost = ga_cost
            ga_result.time_elapsed = ga_time
            ga_result.convergence_history = ga.convergence_history
        
        if n_trials > 1:
            print(f"Cost: {ga_cost:.6f}")
    
    ga_result.compute_convergence_speed()
    results.append(ga_result)
    
    # NOTE: For brevity, remaining algorithms run with n_trials but simplified code
    # All algorithms compute convergence speed
    
    # --- 3. ACOR (Ant Colony Optimization for Continuous) ---
    if ACOR_Solver:
        print("\n[3/4] Running ACOR (Swarm Intelligence)...")
        acor_result = AlgorithmResults(
            name="ACOR (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'archive_size': 30, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            acor = ACOR_Solver(
                cost_function=function,
                n_dims=n_dims,
                bounds=bounds,
                archive_size=30,
                sample_size=50,
                n_iterations=100,
                q=0.5,
                zeta=1.0,
                track_archive=False
            )
            acor_solution, acor_cost = acor.run()
            acor_time = time.time() - start_time
            
            acor_result.add_run(acor_cost, acor_time, acor.convergence_history)
            
            if acor_cost < acor_result.cost:
                acor_result.solution = acor_solution
                acor_result.cost = acor_cost
                acor_result.time_elapsed = acor_time
                acor_result.convergence_history = acor.convergence_history
            
            if n_trials > 1:
                print(f"Cost: {acor_cost:.6f}")
        
        acor_result.compute_convergence_speed()
        results.append(acor_result)
    else:
        print("\n[3/4] ACOR not available - skipping")
    
    # --- 4. PSO (Particle Swarm Optimization) ---
    if PSO:
        print("\n[4/5] Running PSO (Swarm Intelligence)...")
        pso_result = AlgorithmResults(
            name="PSO (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'particles': 20, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            pso = PSO(
                obj_func=function,
                n_particles=20,
                n_dims=n_dims,
                bounds_low=bounds[0],
                bounds_high=bounds[1],
                w=0.8,
                c1=0.5,
                c2=0.5
            )
            pso_solution, pso_cost, pso_history, _, _ = pso.optimize(max_iterations=100)
            pso_time = time.time() - start_time
            
            pso_result.add_run(pso_cost, pso_time, pso_history)
            
            if pso_cost < pso_result.cost:
                pso_result.solution = pso_solution
                pso_result.cost = pso_cost
                pso_result.time_elapsed = pso_time
                pso_result.convergence_history = pso_history
            
            if n_trials > 1:
                print(f"Cost: {pso_cost:.6f}")
        
        pso_result.compute_convergence_speed()
        results.append(pso_result)
    else:
        print("\n[4/5] PSO not available - skipping")
    
    # --- 5. Firefly Algorithm ---
    if FireflyAlgorithm:
        print("\n[5/6] Running Firefly Algorithm (Swarm Intelligence)...")
        fa_result = AlgorithmResults(
            name="Firefly (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'fireflies': 30, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            fa = FireflyAlgorithm(
                objective_func=function,
                dimensions=n_dims,
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                n_fireflies=30,
                max_iterations=100,
                alpha=0.5,
                beta0=1.0,
                gamma=0.01
            )
            fa_solution, fa_cost, fa_history = fa.run()
            fa_time = time.time() - start_time
            
            fa_result.add_run(fa_cost, fa_time, fa_history)
            
            if fa_cost < fa_result.cost:
                fa_result.solution = fa_solution
                fa_result.cost = fa_cost
                fa_result.time_elapsed = fa_time
                fa_result.convergence_history = fa_history
            
            if n_trials > 1:
                print(f"Cost: {fa_cost:.6f}")
        
        fa_result.compute_convergence_speed()
        results.append(fa_result)
    else:
        print("\n[5/6] Firefly Algorithm not available - skipping")

    # --- 6. Artificial Bee Colony ---
    if artificial_bee_colony:
        print("\n[6/7] Running Artificial Bee Colony (Swarm Intelligence)...")
        abc_result = AlgorithmResults(
            name="ABC (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'population': 50, 'iterations': 100, 'limit': 20}
        )
        
        # Configure ABC parameters
        N = 50              # Number of food sources (population size)
        D = n_dims          # Problem dimension
        MaxGen = 100        # Number of iterations
        limit = 20          # Trial limit for scout bees
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            
            try:
                abc_solution, abc_cost, abc_history = artificial_bee_colony(
                    function, bounds[0], bounds[1], D, N, MaxGen, limit
                )
            except Exception as e:
                print(f"Error running ABC continuous: {e}")
                abc_solution, abc_cost, abc_history = None, float('inf'), []
            
            abc_time = time.time() - start_time
            
            abc_result.add_run(abc_cost, abc_time, abc_history)
            
            if abc_cost < abc_result.cost:
                abc_result.solution = abc_solution
                abc_result.cost = abc_cost
                abc_result.time_elapsed = abc_time
                abc_result.convergence_history = abc_history
            
            if n_trials > 1:
                print(f"Cost: {abc_cost:.6f}")
        
        abc_result.compute_convergence_speed()
        results.append(abc_result)
    else:
        print("\n[6/7] Artificial Bee Colony not available - skipping")

    # --- 7. Cuckoo Search ---
    if CS_Solver:
        print("\n[7/7] Running Cuckoo Search (Swarm Intelligence)...")
        cs_result = AlgorithmResults(
            name="CS (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'nests': 50, 'iterations': 100, 'discovery_rate': 0.25}
        )
        
        # Configure CS parameters
        N = 50              # Number of nests (population)
        MaxGen = 100        # Number of generations
        pa = 0.25          # Discovery rate 
        alpha = 0.01       # Step size scaling
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            
            try:
                cs_solution, cs_cost, cs_history = CS_Solver(
                    func_to_optimize=function,
                    LB=bounds[0],
                    UB=bounds[1],
                    D=n_dims,
                    N=N,
                    MaxGen=MaxGen,
                    pa=pa,
                    alpha=alpha
                )
            except Exception as e:
                print(f"Error running CS continuous: {e}")
                cs_solution, cs_cost, cs_history = None, float('inf'), []
            
            cs_time = time.time() - start_time
            
            cs_result.add_run(cs_cost, cs_time, cs_history)
            
            if cs_cost < cs_result.cost:
                cs_result.solution = cs_solution
                cs_result.cost = cs_cost
                cs_result.time_elapsed = cs_time
                cs_result.convergence_history = cs_history
            
            if n_trials > 1:
                print(f"Cost: {cs_cost:.6f}")
        
        cs_result.compute_convergence_speed()
        results.append(cs_result)
    else:
        print("\n[7/7] Cuckoo Search not available - skipping")
    
    return results


def run_tsp_comparison(n_cities=20, map_size=100, seed=42, n_trials=1):
    """
    Compare all algorithms on TSP problem.
    
    Args:
        n_cities: Number of cities
        map_size: Size of the map
        seed: Random seed for reproducibility
        n_trials: Number of independent runs for robustness testing
    """
    print("\n" + "="*80)
    print(f"TSP COMPARISON: {n_cities} Cities")
    if n_trials > 1:
        print(f"Running {n_trials} independent trials for robustness analysis")
    print("="*80)
    
    # Generate cities
    cities = create_cities(n_cities, map_size=map_size, seed=seed)
    
    results = []
    
    # Note: For TSP with multiple trials, we'll run each algorithm once but could extend
    # to multiple trials similar to continuous optimization if needed
    
    # --- 1. Hill Climbing (2-opt) ---
    print("\n[1/5] Running Hill Climbing (2-opt)...")
    hc_result = AlgorithmResults(
        name="Hill Climbing",
        solution=None,
        cost=float('inf'),
        time_elapsed=0,
        convergence_history=[],
        extra_info={'restarts': 10}
    )
    
    for trial in range(n_trials):
        if n_trials > 1:
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
        start_time = time.time()
        hc_tsp = HillClimbingTSP(cities, n_restarts=10, max_iterations=1000)
        hc_tour, hc_cost = hc_tsp.run()
        hc_time = time.time() - start_time
        
        hc_result.add_run(hc_cost, hc_time, hc_tsp.convergence_history)
        
        if hc_cost < hc_result.cost:
            hc_result.solution = hc_tour
            hc_result.cost = hc_cost
            hc_result.time_elapsed = hc_time
            hc_result.convergence_history = hc_tsp.convergence_history
        
        if n_trials > 1:
            print(f"Cost: {hc_cost:.2f}")
    
    hc_result.compute_convergence_speed()
    results.append(hc_result)
    
    # --- 2. A* Search (only for small instances) ---
    if n_cities <= 12:
        print("\n[2/5] Running A* Search...")
        start_time = time.time()
        astar_tsp = AStarTSP(cities, time_limit=60)
        astar_tour, astar_cost = astar_tsp.run()
        astar_time = time.time() - start_time
        
        astar_result = AlgorithmResults(
            name="A* Search",
            solution=astar_tour,
            cost=astar_cost,
            time_elapsed=astar_time,
            convergence_history=[],
            extra_info={'nodes_explored': astar_tsp.nodes_explored}
        )
        astar_result.compute_convergence_speed()
        results.append(astar_result)
    else:
        print(f"\n[2/5] A* Search skipped (n_cities={n_cities} > 12)")
        print("     A* is only practical for small TSP instances")
    
    # --- 3. Genetic Algorithm ---
    print("\n[3/5] Running Genetic Algorithm...")
    ga_result = AlgorithmResults(
        name="Genetic Algorithm",
        solution=None,
        cost=float('inf'),
        time_elapsed=0,
        convergence_history=[],
        extra_info={'population': 100, 'generations': 200}
    )
    
    for trial in range(n_trials):
        if n_trials > 1:
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
        start_time = time.time()
        ga_tsp = GeneticAlgorithmTSP(
            cities,
            population_size=100,
            n_generations=200,
            crossover_rate=0.9,
            mutation_rate=0.2,
            elite_size=5
        )
        ga_tour, ga_cost = ga_tsp.run()
        ga_time = time.time() - start_time
        
        ga_result.add_run(ga_cost, ga_time, ga_tsp.convergence_history)
        
        if ga_cost < ga_result.cost:
            ga_result.solution = ga_tour
            ga_result.cost = ga_cost
            ga_result.time_elapsed = ga_time
            ga_result.convergence_history = ga_tsp.convergence_history
        
        if n_trials > 1:
            print(f"Cost: {ga_cost:.2f}")
    
    ga_result.compute_convergence_speed()
    results.append(ga_result)
    
    # --- 4. ACO (Ant Colony Optimization) ---
    if ACO_TSP_Solver:
        print("\n[4/5] Running ACO (Swarm Intelligence)...")
        aco_result = AlgorithmResults(
            name="ACO (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'ants': 20, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            aco_tsp = ACO_TSP_Solver(
                coordinates=cities,
                n_ants=20,
                n_iterations=100,
                alpha=1.0,
                beta=2.0,
                rho=0.5,
                Q=100
            )
            aco_tour, aco_cost = aco_tsp.run()
            aco_time = time.time() - start_time
            
            # Convert tour format (ACO includes return to start)
            aco_tour_clean = aco_tour[:-1] if aco_tour[-1] == aco_tour[0] else aco_tour
            
            aco_result.add_run(aco_cost, aco_time, aco_tsp.convergence_history)
            
            if aco_cost < aco_result.cost:
                aco_result.solution = aco_tour_clean
                aco_result.cost = aco_cost
                aco_result.time_elapsed = aco_time
                aco_result.convergence_history = aco_tsp.convergence_history
            
            if n_trials > 1:
                print(f"Cost: {aco_cost:.2f}")
        
        aco_result.compute_convergence_speed()
        results.append(aco_result)
    else:
        print("\n[4/5] ACO not available - skipping")
    
    # --- 5. Hybrid PSO ---
    if HybridPSOSolver:
        print("\n[5/6] Running Hybrid PSO (Swarm Intelligence)...")
        pso_result = AlgorithmResults(
            name="Hybrid PSO (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'particles': 20, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            pso_tsp = HybridPSOSolver(
                n_particles=20,
                n_iterations=50,
                w=0.729,
                c1=1.494,
                c2=1.494
            )
            pso_tour, pso_cost, pso_curve = pso_tsp.solve(cities)
            pso_time = time.time() - start_time
            
            pso_result.add_run(pso_cost, pso_time, pso_curve)
            
            if pso_cost < pso_result.cost:
                pso_result.solution = pso_tour
                pso_result.cost = pso_cost
                pso_result.time_elapsed = pso_time
                pso_result.convergence_history = pso_curve
            
            if n_trials > 1:
                print(f"Cost: {pso_cost:.2f}")
        
        pso_result.compute_convergence_speed()
        results.append(pso_result)
    else:
        print("\n[5/6] Hybrid PSO not available - skipping")
    
    # --- 6. Discrete Firefly Algorithm ---
    if DiscreteFireflyAlgorithm:
        print("\n[6/6] Running Discrete Firefly Algorithm (Swarm Intelligence)...")
        fa_result = AlgorithmResults(
            name="Firefly (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'fireflies': 30, 'iterations': 100}
        )
        
        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            dfa = DiscreteFireflyAlgorithm(
                cities=cities,
                n_fireflies=30,
                max_iterations=100,
                alpha=1.0,
                beta0=1.0,
                gamma=0.01
            )
            fa_tour, fa_cost, fa_history = dfa.run()
            fa_time = time.time() - start_time
            
            fa_result.add_run(fa_cost, fa_time, fa_history)
            
            if fa_cost < fa_result.cost:
                fa_result.solution = fa_tour
                fa_result.cost = fa_cost
                fa_result.time_elapsed = fa_time
                fa_result.convergence_history = fa_history
            
            if n_trials > 1:
                print(f"Cost: {fa_cost:.2f}")
        
        fa_result.compute_convergence_speed()
        results.append(fa_result)
    else:
        print("\n[6/6] Discrete Firefly Algorithm not available - skipping")

    # --- 7. Artificial Bee Colony for TSP ---
    if 'artificial_bee_colony_tsp' in globals() and artificial_bee_colony_tsp:
        print("\n[7/8] Running Artificial Bee Colony for TSP (Swarm Intelligence)...")
        
        # Configure ABC parameters
        N = 40              # Number of food sources (population size)
        D = len(cities)     # Dimension (number of cities)
        MaxGen = 100        # Number of iterations
        limit = 20          # Trial limit for scout bees
        
        abc_result = AlgorithmResults(
            name="ABC (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'population': N, 'iterations': MaxGen, 'limit': limit}
        )

        # Create distance matrix (once for all trials)
        dist_matrix = np.zeros((len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            
            try:
                abc_tour, abc_cost, abc_history = artificial_bee_colony_tsp(dist_matrix, D, N, MaxGen, limit)
            except Exception as e:
                print(f"Error running ABC TSP: {e}")
                abc_tour, abc_cost, abc_history = None, float('inf'), []

            abc_time = time.time() - start_time
            
            abc_result.add_run(abc_cost, abc_time, abc_history)
            
            if abc_cost < abc_result.cost:
                abc_result.solution = abc_tour
                abc_result.cost = abc_cost
                abc_result.time_elapsed = abc_time
                abc_result.convergence_history = abc_history
            
            if n_trials > 1:
                print(f"Cost: {abc_cost:.2f}")

        abc_result.compute_convergence_speed()
        results.append(abc_result)
    else:
        print("\n[7/8] Artificial Bee Colony for TSP not available - skipping")

    # --- 8. Cuckoo Search for TSP ---
    if CS_TSP_Solver:
        print("\n[8/8] Running Cuckoo Search for TSP (Swarm Intelligence)...")

        # Configure CS parameters
        N = 40              # Number of nests
        D = len(cities)     # Number of cities
        MaxGen = 100        # Number of iterations
        pa = 0.25          # Discovery rate
        
        cs_result = AlgorithmResults(
            name="CS (Swarm)",
            solution=None,
            cost=float('inf'),
            time_elapsed=0,
            convergence_history=[],
            extra_info={'nests': N, 'iterations': MaxGen, 'discovery_rate': pa}
        )

        # Create distance matrix (once for all trials)
        dist_matrix = np.zeros((len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

        for trial in range(n_trials):
            if n_trials > 1:
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            start_time = time.time()
            
            try:
                cs_tour, cs_cost, cs_history = CS_TSP_Solver(
                    dist_matrix=dist_matrix, 
                    D=D,
                    N=N, 
                    MaxGen=MaxGen, 
                    pa=pa
                )
            except Exception as e:
                print(f"Error running CS TSP: {e}")
                cs_tour, cs_cost, cs_history = None, float('inf'), []

            cs_time = time.time() - start_time
            
            cs_result.add_run(cs_cost, cs_time, cs_history)
            
            if cs_cost < cs_result.cost:
                cs_result.solution = cs_tour
                cs_result.cost = cs_cost
                cs_result.time_elapsed = cs_time
                cs_result.convergence_history = cs_history
            
            if n_trials > 1:
                print(f"Cost: {cs_cost:.2f}")
        
        cs_result.compute_convergence_speed()
        results.append(cs_result)
    else:
        print("\n[8/8] Cuckoo Search for TSP not available - skipping")

    return results, cities


# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

def print_results_table(results, problem_name):
    """Print a formatted results table."""
    print("\n" + "="*80)
    print(f"{problem_name} - RESULTS SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<25} {'Best Cost':<15} {'Time (s)':<12} {'Notes'}")
    print("-"*80)
    
    for result in results:
        notes = ', '.join([f"{k}={v}" for k, v in result.extra_info.items()])
        print(f"{result.name:<25} {result.cost:<15.6f} {result.time_elapsed:<12.3f} {notes}")
    
    # Find best algorithm
    best_result = min(results, key=lambda r: r.cost)
    print("-"*80)
    print(f"Best Algorithm: {best_result.name} (Cost: {best_result.cost:.6f})")
    print("="*80)


def plot_convergence_comparison(results, problem_name, save_path=None):
    """Plot convergence comparison."""
    plt.figure(figsize=(14, 6))
    
    # Separate traditional and swarm
    traditional = [r for r in results if 'Swarm' not in r.name]
    swarm = [r for r in results if 'Swarm' in r.name]
    
    # Plot traditional algorithms
    for result in traditional:
        if len(result.convergence_history) > 0:
            plt.plot(result.convergence_history, label=result.name, 
                    linewidth=2, linestyle='--', alpha=0.7)
    
    # Plot swarm algorithms
    for result in swarm:
        if len(result.convergence_history) > 0:
            plt.plot(result.convergence_history, label=result.name, 
                    linewidth=2.5, alpha=0.9)
    
    plt.xlabel('Iteration / Generation / Restart', fontsize=12)
    plt.ylabel('Best Cost (Fitness)', fontsize=12)
    plt.title(f'{problem_name} - Convergence Comparison\n(Swarm vs Traditional Algorithms)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_performance_bars(results, problem_name, save_path=None):
    """Plot bar charts for cost and time comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [r.name for r in results]
    costs = [r.cost for r in results]
    times = [r.time_elapsed for r in results]
    
    # Color code: traditional vs swarm
    colors = ['steelblue' if 'Swarm' not in name else 'orange' for name in names]
    
    # Cost comparison
    ax1.bar(range(len(names)), costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Best Cost (lower is better)', fontsize=11)
    ax1.set_title('Solution Quality Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (cost, name) in enumerate(zip(costs, names)):
        ax1.text(i, cost, f'{cost:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Time comparison
    ax2.bar(range(len(names)), times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (t, name) in enumerate(zip(times, names)):
        ax2.text(i, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Traditional'),
        Patch(facecolor='orange', alpha=0.7, label='Swarm Intelligence')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10)
    
    fig.suptitle(f'{problem_name} - Performance Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_tsp_solutions(cities, results, save_path=None):
    """Plot TSP solutions for comparison."""
    n_algorithms = len(results)
    cols = 3
    rows = (n_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_algorithms > 1 else [axes]
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        tour = result.solution
        
        # Plot tour
        tour_cities = cities[tour]
        tour_cities = np.vstack([tour_cities, tour_cities[0]])
        ax.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=1.5, alpha=0.7)
        
        # Plot cities
        ax.scatter(cities[:, 0], cities[:, 1], c='red', s=80, zorder=5)
        ax.scatter(cities[tour[0], 0], cities[tour[0], 1], 
                  c='green', s=200, marker='*', zorder=6, label='Start')
        
        ax.set_title(f"{result.name}\nCost: {result.cost:.2f}, Time: {result.time_elapsed:.2f}s",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_algorithms, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'TSP Solutions Comparison ({len(cities)} cities)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# NEW METRICS ANALYSIS AND VISUALIZATION
# ============================================================================

def print_comprehensive_metrics_table(results, problem_name):
    """Print comprehensive metrics table including all new metrics."""
    print("\n" + "="*120)
    print(f"{problem_name} - COMPREHENSIVE METRICS ANALYSIS")
    print("="*120)
    print(f"{'Algorithm':<20} {'Best Cost':<12} {'Time(s)':<10} {'Conv.Speed':<12} {'Robustness':<15} {'Memory(MB)':<12}")
    print("-"*120)
    
    for result in results:
        # Convergence speed
        if result.convergence_speed > 0:
            conv_speed = result.convergence_speed
        elif result.convergence_history:
            conv_speed = len(result.convergence_history)
        else:
            conv_speed = 0  # No convergence data
        
        # Robustness (CV = coefficient of variation)
        robustness_metrics = result.compute_robustness_metrics()
        if robustness_metrics:
            cv = robustness_metrics['coefficient_of_variation']
            robustness_str = f"CV={cv:.4f}"
        else:
            robustness_str = "N/A (1 run)"
        
        # Memory
        mem_str = f"{result.memory_usage:.2f}" if result.memory_usage > 0 else "N/A"
        
        # Display convergence speed
        conv_str = f"{conv_speed}" if conv_speed > 0 else "N/A"
        
        print(f"{result.name:<20} {result.cost:<12.6f} {result.time_elapsed:<10.3f} "
              f"{conv_str:<12} {robustness_str:<15} {mem_str:<12}")
    
    print("="*120)
    print("\nMetric Definitions:")
    print("  - Best Cost: Lowest objective function value achieved")
    print("  - Time(s): Computational time in seconds")
    print("  - Conv.Speed: Iterations needed to reach 90% of final solution quality (lower is faster)")
    print("  - Robustness: Coefficient of Variation (std/mean) across multiple runs (lower is more robust)")
    print("  - Memory(MB): Peak memory usage during execution")
    print("="*120)


def plot_robustness_analysis(results, problem_name, save_path=None):
    """Plot robustness analysis with box plots."""
    # Filter results that have multiple runs
    results_with_trials = [r for r in results if r.multiple_runs]
    
    if not results_with_trials:
        print(f"\n⚠ No multiple-run data available for robustness analysis in {problem_name}.")
        print("  All algorithms ran with n_trials=1. Increase n_trials parameter for robustness analysis.")
        return
    
    print(f"\n📊 Generating robustness analysis for {len(results_with_trials)} algorithm(s) with multiple runs...")
    if len(results_with_trials) < len(results):
        print(f"  Note: {len(results) - len(results_with_trials)} algorithm(s) ran with single trial and are excluded from this plot.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for box plots
    names = [r.name for r in results_with_trials]
    cost_data = [[run['cost'] for run in r.multiple_runs] for r in results_with_trials]
    time_data = [[run['time'] for run in r.multiple_runs] for r in results_with_trials]
    
    # Colors
    colors = ['steelblue' if 'Swarm' not in name else 'orange' for name in names]
    
    # Cost robustness
    bp1 = ax1.boxplot(cost_data, labels=names, patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Solution Quality Robustness', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Time robustness
    bp2 = ax2.boxplot(time_data, labels=names, patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Computational Time Robustness', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add note about number of runs
    n_runs = len(results_with_trials[0].multiple_runs)
    fig.text(0.5, 0.02, f'Note: Box plots show distribution from {n_runs} independent runs. Red diamond = mean.',
             ha='center', fontsize=10, style='italic')
    
    fig.suptitle(f'{problem_name} - Robustness Analysis\n(Box plots from multiple independent runs)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness plot saved to: {save_path}")
    
    plt.show()


def plot_convergence_speed_comparison(results, problem_name, save_path=None):
    """Plot convergence speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r.name for r in results]
    conv_speeds = []
    for r in results:
        if r.convergence_speed > 0:
            conv_speeds.append(r.convergence_speed)
        elif r.convergence_history:
            conv_speeds.append(len(r.convergence_history))
        else:
            conv_speeds.append(0)  # No convergence data
    
    colors = ['steelblue' if 'Swarm' not in name else 'orange' for name in names]
    
    bars = ax.bar(range(len(names)), conv_speeds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Iterations to 90% of Final Solution', fontsize=12)
    ax.set_title(f'{problem_name} - Convergence Speed Comparison\n(Lower is faster, 0 = no convergence tracking)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (speed, name) in enumerate(zip(conv_speeds, names)):
        if speed > 0:
            ax.text(i, speed, f'{speed}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(i, 0.5, 'N/A', ha='center', va='bottom', fontsize=9, style='italic')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Traditional'),
        Patch(facecolor='orange', alpha=0.7, label='Swarm Intelligence')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence speed plot saved to: {save_path}")
    
    plt.show()


def run_scalability_analysis(algorithm_runner, problem_sizes, problem_name, save_path=None):
    """
    Run scalability analysis by testing algorithm performance across different problem sizes.
    
    Args:
        algorithm_runner: Function that takes problem_size and returns results
        problem_sizes: List of problem sizes to test
        problem_name: Name of the problem for plotting
        save_path: Path to save the plot
    """
    print("\n" + "="*80)
    print(f"SCALABILITY ANALYSIS: {problem_name}")
    print("="*80)
    
    scalability_results = {}
    
    for size in problem_sizes:
        print(f"\nTesting problem size: {size}")
        results = algorithm_runner(size)
        scalability_results[size] = results
    
    # Plot scalability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get all algorithm names
    algo_names = list(set(r.name for results in scalability_results.values() for r in results))
    
    for algo_name in algo_names:
        sizes_list = []
        costs_list = []
        times_list = []
        
        for size in problem_sizes:
            results = scalability_results[size]
            algo_result = next((r for r in results if r.name == algo_name), None)
            if algo_result:
                sizes_list.append(size)
                costs_list.append(algo_result.cost)
                times_list.append(algo_result.time_elapsed)
        
        if sizes_list:
            # Cost scaling
            linestyle = '--' if 'Swarm' not in algo_name else '-'
            linewidth = 2 if 'Swarm' in algo_name else 1.5
            ax1.plot(sizes_list, costs_list, marker='o', label=algo_name, 
                    linestyle=linestyle, linewidth=linewidth, markersize=8)
            
            # Time scaling
            ax2.plot(sizes_list, times_list, marker='s', label=algo_name,
                    linestyle=linestyle, linewidth=linewidth, markersize=8)
    
    ax1.set_xlabel('Problem Size', fontsize=12)
    ax1.set_ylabel('Best Cost', fontsize=12)
    ax1.set_title('Solution Quality vs Problem Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Problem Size', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Computational Time vs Problem Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    fig.suptitle(f'{problem_name} - Scalability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scalability plot saved to: {save_path}")
    
    plt.show()
    
    return scalability_results


def print_complexity_analysis(results, problem_name):
    """Print computational complexity analysis."""
    print("\n" + "="*100)
    print(f"{problem_name} - COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("="*100)
    print(f"{'Algorithm':<25} {'Time Complexity':<30} {'Space Complexity':<30}")
    print("-"*100)
    
    complexity_info = {
        'Hill Climbing': ('O(n × k × d)', 'O(d)'),
        'Genetic Algorithm': ('O(g × p × d)', 'O(p × d)'),
        'A* Search': ('O(b^d)', 'O(b^d)'),
        'ACOR (Swarm)': ('O(i × k × d²)', 'O(k × d)'),
        'PSO (Swarm)': ('O(i × p × d)', 'O(p × d)'),
        'Firefly (Swarm)': ('O(i × n² × d)', 'O(n × d)'),
        'ABC (Swarm)': ('O(i × n × d)', 'O(n × d)'),
        'CS (Swarm)': ('O(i × n × d)', 'O(n × d)'),
        'ACO (Swarm)': ('O(i × a × n²)', 'O(a × n)'),
        'Hybrid PSO (Swarm)': ('O(i × p × n²)', 'O(p × n)'),
    }
    
    for result in results:
        time_c, space_c = complexity_info.get(result.name, ('N/A', 'N/A'))
        print(f"{result.name:<25} {time_c:<30} {space_c:<30}")
    
    print("="*100)
    print("\nNotation:")
    print("  n = problem size (dimensions or cities)")
    print("  d = dimensions")
    print("  i = iterations")
    print("  p = population size")
    print("  k = archive/sample size")
    print("  g = generations")
    print("  a = number of ants")
    print("  b = branching factor")
    print("="*100)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    """Run comprehensive comparison with new metrics."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*10 + "COMPREHENSIVE ALGORITHM COMPARISON WITH ADVANCED METRICS" + " "*12 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Configuration
    N_TRIALS = 5  # Number of independent runs for robustness analysis
    
    # ========================================================================
    # CONTINUOUS OPTIMIZATION COMPARISON
    # ========================================================================
    print("\n\n" + "▓"*80)
    print("▓ PART 1: CONTINUOUS OPTIMIZATION (Rastrigin Function)")
    print("▓"*80)
    
    continuous_results = run_continuous_comparison(
        n_dims=5,
        bounds=(-5.12, 5.12),
        function=rastrigin,
        function_name="Rastrigin",
        n_trials=N_TRIALS
    )
    
    # Traditional results table
    print_results_table(continuous_results, "CONTINUOUS OPTIMIZATION (Rastrigin 5D)")
    
    # NEW: Comprehensive metrics table
    print_comprehensive_metrics_table(continuous_results, "CONTINUOUS OPTIMIZATION (Rastrigin 5D)")
    
    # NEW: Complexity analysis
    print_complexity_analysis(continuous_results, "CONTINUOUS OPTIMIZATION")
    
    # Convergence comparison
    plot_convergence_comparison(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_convergence.png"
    )
    
    # Performance bars
    plot_performance_bars(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_bars.png"
    )
    
    # NEW: Convergence speed comparison
    plot_convergence_speed_comparison(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_convergence_speed.png"
    )
    
    # NEW: Robustness analysis
    plot_robustness_analysis(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_robustness.png"
    )
    
    # NEW: Scalability analysis
    print("\n" + "▓"*80)
    print("▓ SCALABILITY ANALYSIS: Testing Different Problem Dimensions")
    print("▓"*80)
    
    def continuous_scalability_runner(n_dims):
        """Helper function for scalability analysis."""
        return run_continuous_comparison(
            n_dims=n_dims,
            bounds=(-5.12, 5.12),
            function=rastrigin,
            function_name="Rastrigin",
            n_trials=1  # Single trial for scalability to save time
        )
    
    scalability_continuous = run_scalability_analysis(
        algorithm_runner=continuous_scalability_runner,
        problem_sizes=[2, 5, 10, 15],
        problem_name="Rastrigin Function - Dimensionality",
        save_path="comparison_rastrigin_scalability.png"
    )
    
    # ========================================================================
    # TSP COMPARISON
    # ========================================================================
    print("\n\n" + "▓"*80)
    print("▓ PART 2: DISCRETE OPTIMIZATION (Traveling Salesman Problem)")
    print("▓"*80)
    n_cities = 30
    tsp_results, cities = run_tsp_comparison(
        n_cities=n_cities,
        map_size=200,
        seed=42,
        n_trials=N_TRIALS
    )
    
    # Traditional results table
    print_results_table(tsp_results, f"TSP ({n_cities} Cities)")
    
    # NEW: Comprehensive metrics table
    print_comprehensive_metrics_table(tsp_results, f"TSP ({n_cities} Cities)")
    
    # NEW: Complexity analysis
    print_complexity_analysis(tsp_results, "TSP DISCRETE OPTIMIZATION")
    
    # Convergence comparison
    plot_convergence_comparison(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_convergence.png"
    )
    
    # Performance bars
    plot_performance_bars(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_bars.png"
    )
    
    # NEW: Convergence speed comparison
    plot_convergence_speed_comparison(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_convergence_speed.png"
    )
    
    # NEW: Robustness analysis
    plot_robustness_analysis(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_robustness.png"
    )
    
    # TSP solutions visualization
    plot_tsp_solutions(
        cities,
        tsp_results,
        save_path="comparison_tsp_solutions.png"
    )
    
    # NEW: TSP Scalability analysis
    print("\n" + "▓"*80)
    print("▓ SCALABILITY ANALYSIS: Testing Different Number of Cities")
    print("▓"*80)
    
    def tsp_scalability_runner(n_cities):
        """Helper function for TSP scalability analysis."""
        results, _ = run_tsp_comparison(
            n_cities=n_cities,
            map_size=200,
            seed=42,
            n_trials=1  # Single trial for scalability
        )
        return results
    
    scalability_tsp = run_scalability_analysis(
        algorithm_runner=tsp_scalability_runner,
        problem_sizes=[10, 20, 30, 40],
        problem_name="TSP - Number of Cities",
        save_path="comparison_tsp_scalability.png"
    )
    
    # ========================================================================
    # FINAL SUMMARY WITH NEW METRICS
    # ========================================================================
    print("\n\n" + "="*100)
    print("COMPREHENSIVE FINAL SUMMARY - KEY INSIGHTS WITH ADVANCED METRICS")
    print("="*100)
    
    print("\n" + "─"*100)
    print("[1] CONTINUOUS OPTIMIZATION - Rastrigin Function")
    print("─"*100)
    
    best_continuous = min(continuous_results, key=lambda r: r.cost)
    fastest_convergence = min(continuous_results, key=lambda r: r.convergence_speed if r.convergence_speed > 0 else float('inf'))
    
    print(f"\n✓ Best Solution Quality:")
    print(f"  Algorithm: {best_continuous.name}")
    print(f"  Cost: {best_continuous.cost:.6f}")
    print(f"  Time: {best_continuous.time_elapsed:.3f}s")
    
    print(f"\n✓ Fastest Convergence:")
    print(f"  Algorithm: {fastest_convergence.name}")
    print(f"  Iterations to 90% solution: {fastest_convergence.convergence_speed}")
    
    # Robustness analysis
    results_with_trials = [r for r in continuous_results if r.multiple_runs]
    if results_with_trials:
        robustness_scores = []
        for r in results_with_trials:
            metrics = r.compute_robustness_metrics()
            if metrics:
                robustness_scores.append((r.name, metrics['coefficient_of_variation']))
        
        if robustness_scores:
            most_robust = min(robustness_scores, key=lambda x: x[1])
            print(f"\n✓ Most Robust (lowest variation across {N_TRIALS} runs):")
            print(f"  Algorithm: {most_robust[0]}")
            print(f"  Coefficient of Variation: {most_robust[1]:.4f}")
    
    # Category comparison
    swarm_continuous = [r for r in continuous_results if 'Swarm' in r.name]
    trad_continuous = [r for r in continuous_results if 'Swarm' not in r.name]
    
    if swarm_continuous and trad_continuous:
        avg_swarm_cost = np.mean([r.cost for r in swarm_continuous])
        avg_trad_cost = np.mean([r.cost for r in trad_continuous])
        avg_swarm_time = np.mean([r.time_elapsed for r in swarm_continuous])
        avg_trad_time = np.mean([r.time_elapsed for r in trad_continuous])
        avg_swarm_conv = np.mean([r.convergence_speed for r in swarm_continuous if r.convergence_speed > 0])
        avg_trad_conv = np.mean([r.convergence_speed for r in trad_continuous if r.convergence_speed > 0])
        
        print(f"\n✓ Category Comparison (Swarm vs Traditional):")
        print(f"  {'Metric':<30} {'Swarm Intelligence':<20} {'Traditional':<20}")
        print(f"  {'-'*30} {'-'*20} {'-'*20}")
        print(f"  {'Average Solution Quality':<30} {avg_swarm_cost:<20.6f} {avg_trad_cost:<20.6f}")
        print(f"  {'Average Computation Time':<30} {avg_swarm_time:<20.3f}s {avg_trad_time:<20.3f}s")
        print(f"  {'Average Convergence Speed':<30} {avg_swarm_conv:<20.1f} {avg_trad_conv:<20.1f}")
        
        if avg_swarm_cost < avg_trad_cost:
            improvement = ((avg_trad_cost - avg_swarm_cost) / avg_trad_cost) * 100
            print(f"\n  → Swarm algorithms achieve {improvement:.1f}% better solution quality on average!")
        else:
            improvement = ((avg_swarm_cost - avg_trad_cost) / avg_swarm_cost) * 100
            print(f"\n  → Traditional algorithms achieve {improvement:.1f}% better solution quality on average!")
    
    print("\n" + "─"*100)
    print("[2] DISCRETE OPTIMIZATION - TSP")
    print("─"*100)
    
    best_tsp = min(tsp_results, key=lambda r: r.cost)
    fastest_tsp = min(tsp_results, key=lambda r: r.convergence_speed if r.convergence_speed > 0 else float('inf'))
    
    print(f"\n✓ Best Solution Quality:")
    print(f"  Algorithm: {best_tsp.name}")
    print(f"  Tour Cost: {best_tsp.cost:.2f}")
    print(f"  Time: {best_tsp.time_elapsed:.3f}s")
    
    print(f"\n✓ Fastest Convergence:")
    print(f"  Algorithm: {fastest_tsp.name}")
    print(f"  Iterations to 90% solution: {fastest_tsp.convergence_speed}")
    
    # Robustness analysis
    results_with_trials_tsp = [r for r in tsp_results if r.multiple_runs]
    if results_with_trials_tsp:
        robustness_scores_tsp = []
        for r in results_with_trials_tsp:
            metrics = r.compute_robustness_metrics()
            if metrics:
                robustness_scores_tsp.append((r.name, metrics['coefficient_of_variation']))
        
        if robustness_scores_tsp:
            most_robust_tsp = min(robustness_scores_tsp, key=lambda x: x[1])
            print(f"\n✓ Most Robust (lowest variation across {N_TRIALS} runs):")
            print(f"  Algorithm: {most_robust_tsp[0]}")
            print(f"  Coefficient of Variation: {most_robust_tsp[1]:.4f}")
    
    # Category comparison
    swarm_tsp = [r for r in tsp_results if 'Swarm' in r.name]
    trad_tsp = [r for r in tsp_results if 'Swarm' not in r.name]
    
    if swarm_tsp and trad_tsp:
        avg_swarm_cost = np.mean([r.cost for r in swarm_tsp])
        avg_trad_cost = np.mean([r.cost for r in trad_tsp])
        avg_swarm_time = np.mean([r.time_elapsed for r in swarm_tsp])
        avg_trad_time = np.mean([r.time_elapsed for r in trad_tsp])
        avg_swarm_conv = np.mean([r.convergence_speed for r in swarm_tsp if r.convergence_speed > 0])
        avg_trad_conv = np.mean([r.convergence_speed for r in trad_tsp if r.convergence_speed > 0])
        
        print(f"\n✓ Category Comparison (Swarm vs Traditional):")
        print(f"  {'Metric':<30} {'Swarm Intelligence':<20} {'Traditional':<20}")
        print(f"  {'-'*30} {'-'*20} {'-'*20}")
        print(f"  {'Average Solution Quality':<30} {avg_swarm_cost:<20.2f} {avg_trad_cost:<20.2f}")
        print(f"  {'Average Computation Time':<30} {avg_swarm_time:<20.3f}s {avg_trad_time:<20.3f}s")
        print(f"  {'Average Convergence Speed':<30} {avg_swarm_conv:<20.1f} {avg_trad_conv:<20.1f}")
        
        if avg_swarm_cost < avg_trad_cost:
            improvement = ((avg_trad_cost - avg_swarm_cost) / avg_trad_cost) * 100
            print(f"\n  → Swarm algorithms achieve {improvement:.1f}% better solution quality on average!")
        else:
            improvement = ((avg_swarm_cost - avg_trad_cost) / avg_swarm_cost) * 100
            print(f"\n  → Traditional algorithms achieve {improvement:.1f}% better solution quality on average!")
    
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*100)
    print("\n📊 All visualizations saved to current directory:")
    print("\n  Continuous Optimization (Rastrigin):")
    print("    ├─ comparison_rastrigin_convergence.png")
    print("    ├─ comparison_rastrigin_bars.png")
    print("    ├─ comparison_rastrigin_convergence_speed.png")
    print("    ├─ comparison_rastrigin_robustness.png")
    print("    └─ comparison_rastrigin_scalability.png")
    print("\n  Discrete Optimization (TSP):")
    print("    ├─ comparison_tsp_convergence.png")
    print("    ├─ comparison_tsp_bars.png")
    print("    ├─ comparison_tsp_convergence_speed.png")
    print("    ├─ comparison_tsp_robustness.png")
    print("    ├─ comparison_tsp_scalability.png")
    print("    └─ comparison_tsp_solutions.png")
    print("\n" + "="*100)
    print("\n📈 New Metrics Included:")
    print("  ✓ Convergence Speed: How fast algorithms reach near-optimal solutions")
    print("  ✓ Computational Complexity: Time and space complexity analysis")
    print("  ✓ Robustness: Performance consistency across multiple independent runs")
    print("  ✓ Scalability: Performance trends with increasing problem size")
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
