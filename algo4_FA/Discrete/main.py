import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.tsp import create_cities
from algo4_FA.Discrete.FA_tsp import DiscreteFireflyAlgorithm


# --- Visualization 1: Best Tours Plot ---
def plot_best_tours():
    print("Running discrete algorithms for tour visualization...")
    
    # Setup problem
    N_CITIES = 25
    cities = create_cities(N_CITIES, seed=None)
    
    # Run DFA
    dfa = DiscreteFireflyAlgorithm(cities, n_fireflies=100, max_iterations=200)
    dfa_tour, dfa_dist, _ = dfa.run()
    
    
    # Plotting
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    
    all_tours = {
        'DFA': (dfa_tour, dfa_dist, ax1),
    }

    for name, (tour, dist, ax) in all_tours.items():
        ax.scatter(cities[:, 0], cities[:, 1], c='red', zorder=2)
        for i in range(N_CITIES):
            ax.text(cities[i, 0] + 0.5, cities[i, 1] + 0.5, str(i), fontsize=9)
        
        # Draw the tour
        tour_coords = cities[np.append(tour, tour)] # Append start to end
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='blue', zorder=1)
        ax.set_title(f"{name} - Best Tour (Distance: {dist:.2f})")
        
    plt.tight_layout()
    plt.savefig("best_tours_comparison.png")
    plt.show()
    print("Best tour plot saved as 'best_tours_comparison.png'")

# plot_best_tours()
# --- Visualization 2: Discrete Convergence ---
def plot_discrete_convergence():
    print("Running discrete algorithms for convergence comparison...")
    
    # Setup problem
    cities = create_cities(25, seed=None)
    FE_BUDGET = 20000
    
    # Run DFA
    dfa = DiscreteFireflyAlgorithm(cities, n_fireflies=100, max_iterations=200)
    _, _, dfa_history = dfa.run()
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0, FE_BUDGET, len(dfa_history)), dfa_history, label="Discrete FA (DFA)")
    
    plt.title('Algorithm Convergence Comparison on TSP (25 Cities)')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Tour Distance Found')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("convergence_discrete.png")
    plt.show()
    print("Discrete convergence plot saved as 'convergence_discrete.png'")

# plot_discrete_convergence()

def main():
    plot_best_tours()
    plot_discrete_convergence()
if __name__ == "__main__":
    main()
