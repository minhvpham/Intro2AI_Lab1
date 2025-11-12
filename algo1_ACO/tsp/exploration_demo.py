"""
Exploration Demo: Complex TSP Problem
======================================
This file demonstrates how HIGH exploration (low alpha, low beta, high rho) 
allows ACO to better handle complex TSP problems by avoiding local optima.

Core Concept:
- Low α (pheromone weight): Don't over-trust learned paths
- Low β (heuristic weight): Don't be too greedy
- High ρ (evaporation): Quickly forget old paths, try new ones

Expected Behavior:
- Slower convergence
- More diverse solution search
- Better chance of finding global optimum on complex problems
- Avoids premature convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algo1_ACO.tsp.ACO import ACO_TSP_Solver
from utils.tsp import create_cities


def run_exploration_demo():
    """
    Run ACO on a COMPLEX TSP problem (30 cities) with high exploration settings.
    """
    print("=" * 70)
    print("EXPLORATION DEMO: Complex TSP Problem")
    print("=" * 70)
    print("\nScenario: 30 cities - Complex problem with many local optima")
    print("Strategy: High Exploration (Diverse search, avoid premature convergence)")
    print("\nParameter Settings:")
    
    # Complex problem: more cities
    n_cities = 30
    coordinates = create_cities(n_cities, map_size=100, seed=123)
    
    # EXPLORATION SETTINGS
    exploration_params = {
        'n_ants': 30,
        'n_iterations': 100,
        'alpha': 0.5,    # LOW: Don't over-trust pheromone trails
        'beta': 1.0,     # LOW: Don't be too greedy
        'rho': 0.7,      # HIGH: Fast evaporation (forget quickly, explore more)
        'Q': 100
    }
    
    print(f"  α (alpha) = {exploration_params['alpha']:.1f} [LOW - Don't over-trust learned paths]")
    print(f"  β (beta)  = {exploration_params['beta']:.1f} [LOW - Reduce greediness]")
    print(f"  ρ (rho)   = {exploration_params['rho']:.1f} [HIGH - Fast evaporation]")
    print(f"  Ants      = {exploration_params['n_ants']}")
    print(f"  Iterations= {exploration_params['n_iterations']}")
    
    print("\n" + "-" * 70)
    print("Running ACO with HIGH EXPLORATION...")
    print("-" * 70 + "\n")
    
    # Run ACO
    aco = ACO_TSP_Solver(coordinates, **exploration_params)
    best_path, best_length = aco.run()
    
    print("\n" + "=" * 70)
    print(f"RESULT: Best tour length = {best_length:.2f}")
    print("=" * 70)
    
    # For comparison, also run with exploitation settings
    print("\n" + "-" * 70)
    print("Running comparison: EXPLOITATION settings on same problem...")
    print("-" * 70 + "\n")
    
    exploitation_params = {
        'n_ants': 30,
        'n_iterations': 100,
        'alpha': 2.0,
        'beta': 5.0,
        'rho': 0.1,
        'Q': 100
    }
    
    aco_exploit = ACO_TSP_Solver(coordinates, **exploitation_params)
    exploit_path, exploit_length = aco_exploit.run()
    
    print("\n" + "=" * 70)
    print(f"EXPLOITATION RESULT: Best tour length = {exploit_length:.2f}")
    print("=" * 70)
    
    # Create visualization
    create_exploration_visualization(aco, aco_exploit, coordinates, exploration_params, exploitation_params)
    
    return aco, aco_exploit, best_path, best_length


def create_exploration_visualization(aco_explore, aco_exploit, coordinates, explore_params, exploit_params):
    """Create a comprehensive visualization comparing exploration vs exploitation."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Convergence Comparison (Top Left)
    ax1 = plt.subplot(3, 3, 1)
    iterations_explore = range(1, len(aco_explore.convergence_history) + 1)
    iterations_exploit = range(1, len(aco_exploit.convergence_history) + 1)
    
    ax1.plot(iterations_explore, aco_explore.convergence_history, 'b-', linewidth=2.5, 
            label=f'EXPLORATION (final: {aco_explore.best_path_length:.2f})', marker='o', markersize=3)
    ax1.plot(iterations_exploit, aco_exploit.convergence_history, 'r-', linewidth=2.5, 
            label=f'EXPLOITATION (final: {aco_exploit.best_path_length:.2f})', marker='s', markersize=3)
    
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Best Path Length', fontsize=11, fontweight='bold')
    ax1.set_title('Convergence Comparison\nExploration vs Exploitation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Highlight winner
    if aco_explore.best_path_length < aco_exploit.best_path_length:
        ax1.text(0.5, 0.95, '🏆 EXPLORATION WINS!', transform=ax1.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax1.text(0.5, 0.95, '🏆 EXPLOITATION WINS!', transform=ax1.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 2. Exploration Tour (Top Middle)
    ax2 = plt.subplot(3, 3, 2)
    path = aco_explore.best_path
    for i in range(len(path) - 1):
        city_a = path[i]
        city_b = path[i + 1]
        ax2.plot([coordinates[city_a, 0], coordinates[city_b, 0]],
                [coordinates[city_a, 1], coordinates[city_b, 1]],
                'b-', linewidth=1.5, alpha=0.6)
    
    ax2.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=80, zorder=5, 
               edgecolors='black', linewidths=1, alpha=0.7)
    
    start_city = path[0]
    ax2.scatter([coordinates[start_city, 0]], [coordinates[start_city, 1]], 
               c='lime', s=200, marker='*', zorder=6, edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
    ax2.set_title(f'EXPLORATION Tour\nLength: {aco_explore.best_path_length:.2f}', 
                 fontsize=12, fontweight='bold', color='blue')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Exploitation Tour (Top Right)
    ax3 = plt.subplot(3, 3, 3)
    path = aco_exploit.best_path
    for i in range(len(path) - 1):
        city_a = path[i]
        city_b = path[i + 1]
        ax3.plot([coordinates[city_a, 0], coordinates[city_b, 0]],
                [coordinates[city_a, 1], coordinates[city_b, 1]],
                'r-', linewidth=1.5, alpha=0.6)
    
    ax3.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=80, zorder=5, 
               edgecolors='black', linewidths=1, alpha=0.7)
    
    start_city = path[0]
    ax3.scatter([coordinates[start_city, 0]], [coordinates[start_city, 1]], 
               c='yellow', s=200, marker='*', zorder=6, edgecolors='black', linewidths=2)
    
    ax3.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
    ax3.set_title(f'EXPLOITATION Tour\nLength: {aco_exploit.best_path_length:.2f}', 
                 fontsize=12, fontweight='bold', color='red')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Diversity Analysis (Middle Left)
    ax4 = plt.subplot(3, 3, 4)
    
    # Calculate diversity: standard deviation of iteration best lengths
    window = 10
    explore_diversity = []
    exploit_diversity = []
    
    for i in range(window, len(aco_explore.iteration_best_lengths)):
        window_data = aco_explore.iteration_best_lengths[i-window:i]
        explore_diversity.append(np.std(window_data))
    
    for i in range(window, len(aco_exploit.iteration_best_lengths)):
        window_data = aco_exploit.iteration_best_lengths[i-window:i]
        exploit_diversity.append(np.std(window_data))
    
    ax4.plot(range(window, len(aco_explore.iteration_best_lengths)), explore_diversity, 
            'b-', linewidth=2, label='EXPLORATION', alpha=0.7)
    ax4.plot(range(window, len(aco_exploit.iteration_best_lengths)), exploit_diversity, 
            'r-', linewidth=2, label='EXPLOITATION', alpha=0.7)
    
    ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Solution Diversity (Std Dev)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Solution Diversity Over Time\n(Window={window})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Pheromone Comparison (Middle Middle & Right)
    ax5 = plt.subplot(3, 3, 5)
    im1 = ax5.imshow(aco_explore.pheromones, cmap='Blues', interpolation='nearest')
    ax5.set_xlabel('City', fontsize=11, fontweight='bold')
    ax5.set_ylabel('City', fontsize=11, fontweight='bold')
    ax5.set_title('EXPLORATION Pheromones\n(More uniform)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax5, label='Pheromone Level')
    
    ax6 = plt.subplot(3, 3, 6)
    im2 = ax6.imshow(aco_exploit.pheromones, cmap='Reds', interpolation='nearest')
    ax6.set_xlabel('City', fontsize=11, fontweight='bold')
    ax6.set_ylabel('City', fontsize=11, fontweight='bold')
    ax6.set_title('EXPLOITATION Pheromones\n(More concentrated)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax6, label='Pheromone Level')
    
    # 6. Parameter Comparison (Bottom Left)
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    comparison_text = f"""
EXPLORATION vs EXPLOITATION
{'=' * 45}

🌊 EXPLORATION (Complex Problem)
   α = {explore_params['alpha']:.1f}  (LOW - less pheromone trust)
   β = {explore_params['beta']:.1f}  (LOW - less greedy)
   ρ = {explore_params['rho']:.1f}  (HIGH - fast forgetting)
   
   Result: {aco_explore.best_path_length:.2f}
   Strategy: Diverse search, avoid local optima

🔥 EXPLOITATION (Same Problem)
   α = {exploit_params['alpha']:.1f}  (HIGH - strong pheromone trust)
   β = {exploit_params['beta']:.1f}  (HIGH - very greedy)
   ρ = {exploit_params['rho']:.1f}  (LOW - slow forgetting)
   
   Result: {aco_exploit.best_path_length:.2f}
   Strategy: Fast convergence, risk stagnation

{'=' * 45}
Improvement: {((aco_exploit.best_path_length - aco_explore.best_path_length) / aco_exploit.best_path_length * 100):.2f}%
"""
    
    ax7.text(0.05, 0.95, comparison_text, transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7. Improvement Rate Comparison (Bottom Middle)
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate improvement rates
    explore_rates = [0]
    exploit_rates = [0]
    
    for i in range(1, len(aco_explore.convergence_history)):
        rate = (aco_explore.convergence_history[i-1] - aco_explore.convergence_history[i])
        explore_rates.append(max(0, rate))
    
    for i in range(1, len(aco_exploit.convergence_history)):
        rate = (aco_exploit.convergence_history[i-1] - aco_exploit.convergence_history[i])
        exploit_rates.append(max(0, rate))
    
    ax8.plot(explore_rates, 'b-', linewidth=2, label='EXPLORATION', alpha=0.7)
    ax8.plot(exploit_rates, 'r-', linewidth=2, label='EXPLOITATION', alpha=0.7)
    
    ax8.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Absolute Improvement', fontsize=11, fontweight='bold')
    ax8.set_title('Improvement per Iteration', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # 8. Convergence Speed (Bottom Right)
    ax9 = plt.subplot(3, 3, 9)
    
    # Find when each converged (within 1% of final)
    explore_final = aco_explore.convergence_history[-1]
    exploit_final = aco_exploit.convergence_history[-1]
    
    explore_converged = next((i for i, v in enumerate(aco_explore.convergence_history) 
                             if abs(v - explore_final) < 0.01 * explore_final), 
                            len(aco_explore.convergence_history))
    exploit_converged = next((i for i, v in enumerate(aco_exploit.convergence_history) 
                             if abs(v - exploit_final) < 0.01 * exploit_final), 
                            len(aco_exploit.convergence_history))
    
    strategies = ['EXPLORATION', 'EXPLOITATION']
    convergence_iters = [explore_converged + 1, exploit_converged + 1]
    final_values = [aco_explore.best_path_length, aco_exploit.best_path_length]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax9_twin = ax9.twinx()
    
    bars1 = ax9.bar(x - width/2, convergence_iters, width, label='Conv. Iteration', 
                   color=['blue', 'red'], alpha=0.6, edgecolor='black')
    bars2 = ax9_twin.bar(x + width/2, final_values, width, label='Final Length', 
                        color=['lightblue', 'lightcoral'], alpha=0.6, edgecolor='black')
    
    ax9.set_ylabel('Convergence Iteration', fontsize=11, fontweight='bold', color='black')
    ax9_twin.set_ylabel('Final Tour Length', fontsize=11, fontweight='bold', color='black')
    ax9.set_title('Convergence Speed & Quality', fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(strategies, fontweight='bold')
    ax9.legend(loc='upper left', fontsize=9)
    ax9_twin.legend(loc='upper right', fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, convergence_iters)):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    for i, (bar, val) in enumerate(zip(bars2, final_values)):
        height = bar.get_height()
        ax9_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('ACO EXPLORATION DEMO: Complex TSP (30 Cities)\nLow α, Low β, High ρ → Better Escape from Local Optima', 
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'exploration_demo_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "🌊" * 35)
    print("ACO EXPLORATION DEMONSTRATION")
    print("🌊" * 35 + "\n")
    
    aco_explore, aco_exploit, best_path, best_length = run_exploration_demo()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("✓ High exploration → Better handling of complex problems")
    print("✓ Algorithm avoids premature convergence to local optima")
    print("✓ More diverse search leads to better global solutions")
    print("✓ Trade-off: Slower convergence, needs more iterations")
    print("=" * 70)
    
    improvement = ((aco_exploit.best_path_length - aco_explore.best_path_length) / 
                   aco_exploit.best_path_length * 100)
    
    if improvement > 0:
        print(f"\nEXPLORATION achieved {improvement:.2f}% better solution than EXPLOITATION!")
        print("   This demonstrates the value of exploration on complex problems.")
    else:
        print(f"\n EXPLOITATION achieved better solution this run ({-improvement:.2f}%)")
        print("   This can happen due to randomness or problem structure.")
    
    print("=" * 70 + "\n")
