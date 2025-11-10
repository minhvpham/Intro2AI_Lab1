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

class AlgorithmResults:
    """Store results from an algorithm run."""
    def __init__(self, name, solution, cost, time_elapsed, convergence_history=None, extra_info=None):
        self.name = name
        self.solution = solution
        self.cost = cost
        self.time_elapsed = time_elapsed
        self.convergence_history = convergence_history if convergence_history else []
        self.extra_info = extra_info if extra_info else {}


def run_continuous_comparison(n_dims=5, bounds=(-5.12, 5.12), 
                               function=rastrigin, function_name="Rastrigin"):
    """
    Compare all algorithms on continuous optimization problem.
    """
    print("\n" + "="*80)
    print(f"CONTINUOUS OPTIMIZATION COMPARISON: {function_name} Function ({n_dims}D)")
    print("="*80)
    
    results = []
    
    # --- 1. Hill Climbing ---
    print("\n[1/4] Running Hill Climbing...")
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
    
    results.append(AlgorithmResults(
        name="Hill Climbing",
        solution=hc_solution,
        cost=hc_cost,
        time_elapsed=hc_time,
        convergence_history=hc.convergence_history,
        extra_info={'restarts': 10}
    ))
    
    # --- 2. Genetic Algorithm ---
    print("\n[2/4] Running Genetic Algorithm...")
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
    
    results.append(AlgorithmResults(
        name="Genetic Algorithm",
        solution=ga_solution,
        cost=ga_cost,
        time_elapsed=ga_time,
        convergence_history=ga.convergence_history,
        extra_info={'population': 50, 'generations': 100}
    ))
    
    # --- 3. ACOR (Ant Colony Optimization for Continuous) ---
    if ACOR_Solver:
        print("\n[3/4] Running ACOR (Swarm Intelligence)...")
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
        
        results.append(AlgorithmResults(
            name="ACOR (Swarm)",
            solution=acor_solution,
            cost=acor_cost,
            time_elapsed=acor_time,
            convergence_history=acor.convergence_history,
            extra_info={'archive_size': 30, 'iterations': 100}
        ))
    else:
        print("\n[3/4] ACOR not available - skipping")
    
    # --- 4. PSO (Particle Swarm Optimization) ---
    if PSO:
        print("\n[4/5] Running PSO (Swarm Intelligence)...")
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
        
        results.append(AlgorithmResults(
            name="PSO (Swarm)",
            solution=pso_solution,
            cost=pso_cost,
            time_elapsed=pso_time,
            convergence_history=pso_history,
            extra_info={'particles': 20, 'iterations': 100}
        ))
    else:
        print("\n[4/5] PSO not available - skipping")
    
    # --- 5. Firefly Algorithm ---
    if FireflyAlgorithm:
        print("\n[5/6] Running Firefly Algorithm (Swarm Intelligence)...")
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
        
        results.append(AlgorithmResults(
            name="Firefly (Swarm)",
            solution=fa_solution,
            cost=fa_cost,
            time_elapsed=fa_time,
            convergence_history=fa_history,
            extra_info={'fireflies': 30, 'iterations': 100}
        ))
    else:
        print("\n[5/6] Firefly Algorithm not available - skipping")

    # --- 6. Artificial Bee Colony ---
    if artificial_bee_colony:
        print("\n[6/7] Running Artificial Bee Colony (Swarm Intelligence)...")
        start_time = time.time()

        # Configure ABC parameters similar to the implementation
        N = 50              # Number of food sources (population size)
        D = n_dims          # Problem dimension
        MaxGen = 100        # Number of iterations
        limit = 20          # Trial limit for scout bees

        try:
            # artificial_bee_colony expects: func_to_optimize, LB, UB, D, N, MaxGen, limit
            abc_solution, abc_cost, abc_history = artificial_bee_colony(
                function, bounds[0], bounds[1], D, N, MaxGen, limit
            )
        except Exception as e:
            print(f"Error running ABC continuous: {e}")
            abc_solution, abc_cost, abc_history = None, float('inf'), []

        abc_time = time.time() - start_time

        results.append(AlgorithmResults(
            name="ABC (Swarm)",
            solution=abc_solution,
            cost=abc_cost,
            time_elapsed=abc_time,
            convergence_history=abc_history,
            extra_info={'population': N, 'iterations': MaxGen, 'limit': limit}
        ))
    else:
        print("\n[6/7] Artificial Bee Colony not available - skipping")

    # --- 7. Cuckoo Search ---
    if CS_Solver:
        print("\n[7/7] Running Cuckoo Search (Swarm Intelligence)...")
        start_time = time.time()
        
        # Configure CS parameters similar to the implementation
        N = 50              # Number of nests (population)
        MaxGen = 100        # Number of generations
        pa = 0.25          # Discovery rate 
        alpha = 0.01       # Step size scaling

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

        results.append(AlgorithmResults(
            name="CS (Swarm)",
            solution=cs_solution,
            cost=cs_cost,
            time_elapsed=cs_time,
            convergence_history=cs_history,
            extra_info={'nests': N, 'iterations': MaxGen, 'discovery_rate': pa}
        ))
    else:
        print("\n[7/7] Cuckoo Search not available - skipping")
    
    return results


def run_tsp_comparison(n_cities=20, map_size=100, seed=42):
    """
    Compare all algorithms on TSP problem.
    """
    print("\n" + "="*80)
    print(f"TSP COMPARISON: {n_cities} Cities")
    print("="*80)
    
    # Generate cities
    cities = create_cities(n_cities, map_size=map_size, seed=seed)
    
    results = []
    
    # --- 1. Hill Climbing (2-opt) ---
    print("\n[1/5] Running Hill Climbing (2-opt)...")
    start_time = time.time()
    hc_tsp = HillClimbingTSP(cities, n_restarts=10, max_iterations=1000)
    hc_tour, hc_cost = hc_tsp.run()
    hc_time = time.time() - start_time
    
    results.append(AlgorithmResults(
        name="Hill Climbing",
        solution=hc_tour,
        cost=hc_cost,
        time_elapsed=hc_time,
        convergence_history=hc_tsp.convergence_history,
        extra_info={'restarts': 10}
    ))
    
    # --- 2. A* Search (only for small instances) ---
    if n_cities <= 12:
        print("\n[2/5] Running A* Search...")
        start_time = time.time()
        astar_tsp = AStarTSP(cities, time_limit=60)
        astar_tour, astar_cost = astar_tsp.run()
        astar_time = time.time() - start_time
        
        results.append(AlgorithmResults(
            name="A* Search",
            solution=astar_tour,
            cost=astar_cost,
            time_elapsed=astar_time,
            convergence_history=[],
            extra_info={'nodes_explored': astar_tsp.nodes_explored}
        ))
    else:
        print(f"\n[2/5] A* Search skipped (n_cities={n_cities} > 12)")
        print("     A* is only practical for small TSP instances")
    
    # --- 3. Genetic Algorithm ---
    print("\n[3/5] Running Genetic Algorithm...")
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
    
    results.append(AlgorithmResults(
        name="Genetic Algorithm",
        solution=ga_tour,
        cost=ga_cost,
        time_elapsed=ga_time,
        convergence_history=ga_tsp.convergence_history,
        extra_info={'population': 100, 'generations': 200}
    ))
    
    # --- 4. ACO (Ant Colony Optimization) ---
    if ACO_TSP_Solver:
        print("\n[4/5] Running ACO (Swarm Intelligence)...")
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
        
        results.append(AlgorithmResults(
            name="ACO (Swarm)",
            solution=aco_tour_clean,
            cost=aco_cost,
            time_elapsed=aco_time,
            convergence_history=aco_tsp.convergence_history,
            extra_info={'ants': 20, 'iterations': 100}
        ))
    else:
        print("\n[4/5] ACO not available - skipping")
    
    # --- 5. Hybrid PSO ---
    if HybridPSOSolver:
        print("\n[5/6] Running Hybrid PSO (Swarm Intelligence)...")
        start_time = time.time()
        pso_tsp = HybridPSOSolver(
            n_particles=20,
            n_iterations=100,
            w=0.729,
            c1=1.494,
            c2=1.494
        )
        pso_tour, pso_cost, pso_curve = pso_tsp.solve(cities)
        pso_time = time.time() - start_time
        
        results.append(AlgorithmResults(
            name="Hybrid PSO (Swarm)",
            solution=pso_tour,
            cost=pso_cost,
            time_elapsed=pso_time,
            convergence_history=pso_curve,
            extra_info={'particles': 20, 'iterations': 100}
        ))
    else:
        print("\n[5/6] Hybrid PSO not available - skipping")
    
    # --- 6. Discrete Firefly Algorithm ---
    if DiscreteFireflyAlgorithm:
        print("\n[6/6] Running Discrete Firefly Algorithm (Swarm Intelligence)...")
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
        
        results.append(AlgorithmResults(
            name="Firefly (Swarm)",
            solution=fa_tour,
            cost=fa_cost,
            time_elapsed=fa_time,
            convergence_history=fa_history,
            extra_info={'fireflies': 30, 'iterations': 100}
        ))
    else:
        print("\n[6/6] Discrete Firefly Algorithm not available - skipping")

    # --- 7. Artificial Bee Colony for TSP ---
    if 'artificial_bee_colony_tsp' in globals() and artificial_bee_colony_tsp:
        print("\n[7/8] Running Artificial Bee Colony for TSP (Swarm Intelligence)...")
        start_time = time.time()
        # Configure ABC parameters
        N = 40              # Number of food sources (population size)
        D = len(cities)     # Dimension (number of cities)
        MaxGen = 100        # Number of iterations
        limit = 20          # Trial limit for scout bees

        # Create distance matrix
        dist_matrix = np.zeros((len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

        try:
            abc_tour, abc_cost, abc_history = artificial_bee_colony_tsp(dist_matrix, D, N, MaxGen, limit)
        except Exception as e:
            print(f"Error running ABC TSP: {e}")
            abc_tour, abc_cost, abc_history = None, float('inf'), []

        abc_time = time.time() - start_time

        results.append(AlgorithmResults(
            name="ABC (Swarm)",
            solution=abc_tour,
            cost=abc_cost,
            time_elapsed=abc_time,
            convergence_history=abc_history,
            extra_info={'population': N, 'iterations': MaxGen, 'limit': limit}
        ))
    else:
        print("\n[7/8] Artificial Bee Colony for TSP not available - skipping")

    # --- 8. Cuckoo Search for TSP ---
    if CS_TSP_Solver:
        print("\n[8/8] Running Cuckoo Search for TSP (Swarm Intelligence)...")
        start_time = time.time()

        # Configure CS parameters
        N = 40              # Number of nests
        D = len(cities)     # Number of cities
        MaxGen = 100        # Number of iterations
        pa = 0.25          # Discovery rate

        # Create distance matrix
        dist_matrix = np.zeros((len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

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

        results.append(AlgorithmResults(
            name="CS (Swarm)",
            solution=cs_tour,
            cost=cs_cost,
            time_elapsed=cs_time,
            convergence_history=cs_history,
            extra_info={'nests': N, 'iterations': MaxGen, 'discovery_rate': pa}
        ))
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
# MAIN COMPARISON
# ============================================================================

def main():
    """Run comprehensive comparison."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*15 + "SWARM INTELLIGENCE VS TRADITIONAL ALGORITHMS" + " "*17 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
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
        function_name="Rastrigin"
    )
    
    print_results_table(continuous_results, "CONTINUOUS OPTIMIZATION (Rastrigin 5D)")
    
    plot_convergence_comparison(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_convergence.png"
    )
    
    plot_performance_bars(
        continuous_results,
        "Rastrigin Function (5D)",
        save_path="comparison_rastrigin_bars.png"
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
        seed=None
    )
    
    print_results_table(tsp_results, f"TSP ({n_cities} Cities)")
    
    plot_convergence_comparison(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_convergence.png"
    )
    
    plot_performance_bars(
        tsp_results,
        f"TSP ({n_cities} Cities)",
        save_path="comparison_tsp_bars.png"
    )
    
    plot_tsp_solutions(
        cities,
        tsp_results,
        save_path="comparison_tsp_solutions.png"
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("FINAL SUMMARY - KEY INSIGHTS")
    print("="*80)
    
    print("\n[Continuous Optimization - Rastrigin Function]")
    best_continuous = min(continuous_results, key=lambda r: r.cost)
    print(f"  ✓ Best Algorithm: {best_continuous.name}")
    print(f"  ✓ Best Cost: {best_continuous.cost:.6f}")
    print(f"  ✓ Time: {best_continuous.time_elapsed:.3f}s")
    
    swarm_continuous = [r for r in continuous_results if 'Swarm' in r.name]
    trad_continuous = [r for r in continuous_results if 'Swarm' not in r.name]
    
    if swarm_continuous and trad_continuous:
        avg_swarm_cost = np.mean([r.cost for r in swarm_continuous])
        avg_trad_cost = np.mean([r.cost for r in trad_continuous])
        print(f"\n  Average Performance:")
        print(f"    - Swarm Intelligence: {avg_swarm_cost:.6f}")
        print(f"    - Traditional: {avg_trad_cost:.6f}")
        
        if avg_swarm_cost < avg_trad_cost:
            improvement = ((avg_trad_cost - avg_swarm_cost) / avg_trad_cost) * 100
            print(f"    → Swarm algorithms {improvement:.1f}% better on average!")
        else:
            improvement = ((avg_swarm_cost - avg_trad_cost) / avg_swarm_cost) * 100
            print(f"    → Traditional algorithms {improvement:.1f}% better on average!")
    
    print("\n[Discrete Optimization - TSP]")
    best_tsp = min(tsp_results, key=lambda r: r.cost)
    print(f" Best Algorithm: {best_tsp.name}")
    print(f" Best Tour Cost: {best_tsp.cost:.2f}")
    print(f" Time: {best_tsp.time_elapsed:.3f}s")
    
    swarm_tsp = [r for r in tsp_results if 'Swarm' in r.name]
    trad_tsp = [r for r in tsp_results if 'Swarm' not in r.name]
    
    if swarm_tsp and trad_tsp:
        avg_swarm_cost = np.mean([r.cost for r in swarm_tsp])
        avg_trad_cost = np.mean([r.cost for r in trad_tsp])
        print(f"\n  Average Performance:")
        print(f"    - Swarm Intelligence: {avg_swarm_cost:.2f}")
        print(f"    - Traditional: {avg_trad_cost:.2f}")
        
        if avg_swarm_cost < avg_trad_cost:
            improvement = ((avg_trad_cost - avg_swarm_cost) / avg_trad_cost) * 100
            print(f"    → Swarm algorithms {improvement:.1f}% better on average!")
        else:
            improvement = ((avg_swarm_cost - avg_trad_cost) / avg_swarm_cost) * 100
            print(f"    → Traditional algorithms {improvement:.1f}% better on average!")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\nAll plots saved to current directory:")
    print("  - comparison_rastrigin_convergence.png")
    print("  - comparison_rastrigin_bars.png")
    print("  - comparison_tsp_convergence.png")
    print("  - comparison_tsp_bars.png")
    print("  - comparison_tsp_solutions.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
