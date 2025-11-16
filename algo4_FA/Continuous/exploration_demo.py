"""
Exploration Demo: Firefly Algorithm on Complex Multimodal Function (Rastrigin)

This demo demonstrates how FA excels at EXPLORATION on complex, multimodal problems
by using parameters that promote global search and avoid premature convergence:
- Low gamma (γ ∈ [0.001, 0.1]): High visibility, global awareness
- Moderate to high alpha (α ∈ [0.1, 0.5]): Larger random steps for escaping local optima

The Rastrigin function has many local optima, making it ideal for testing exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algo4_FA.Continuous.FA import FireflyAlgorithm


def rastrigin_function(X):
    """
    Rastrigin function (highly multimodal, non-convex).
    Global minimum: f(0,...,0) = 0
    Domain: typically [-5.12, 5.12]^d
    
    This is one of the most challenging benchmark functions - it has:
    - Many local optima (approximately 10^d local minima)
    - Periodic structure with cosine modulation
    - Large basins of attraction for local optima
    - Global optimum at the center
    
    Ideal for testing EXPLORATION capabilities and avoiding premature convergence.
    """
    A = 10
    n_dims = len(X)
    return A * n_dims + np.sum(X**2 - A * np.cos(2 * np.pi * X))


def ackley_function(X):
    """
    Ackley function (highly multimodal, non-convex).
    Global minimum: f(0,...,0) = 0
    Domain: typically [-5, 5]^d
    
    Another challenging multimodal function with:
    - Many local optima
    - Nearly flat outer region
    - Deep global optimum at center
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(X)
    
    sum1 = np.sum(X**2)
    sum2 = np.sum(np.cos(c * X))
    
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)


def run_exploration_experiment():
    """
    Compare FA with different parameter settings on Rastrigin function.
    
    We test three configurations:
    1. High Exploration (γ=0.01, α=0.4): Optimized for multimodal problems
    2. Balanced (γ=0.5, α=0.2): Middle ground
    3. High Exploitation (γ=5.0, α=0.05): Better for unimodal (poor here)
    """
    print("="*70)
    print("EXPLORATION DEMO: Firefly Algorithm on Rastrigin Function")
    print("="*70)
    print("\nThe Rastrigin function is HIGHLY MULTIMODAL with many local optima.")
    print("We expect HIGH EXPLORATION settings to find the global optimum best.\n")
    
    # Problem setup
    dimensions = 10
    lower_bound = -5.12
    upper_bound = 5.12
    n_fireflies = 50
    max_iterations = 200
    
    # Configuration 1: HIGH EXPLORATION (optimal for Rastrigin)
    print("\n" + "-"*70)
    print("Configuration 1: HIGH EXPLORATION")
    print(f"  γ (gamma) = 0.01  [Very low - global visibility, swarm cohesion]")
    print(f"  α (alpha) = 0.4   [High - large random steps, escape local optima]")
    print(f"  β₀ (beta0) = 1.0  [Standard attractiveness]")
    print("-"*70)
    
    fa_explore = FireflyAlgorithm(
        objective_func=rastrigin_function,
        dimensions=dimensions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_fireflies=n_fireflies,
        max_iterations=max_iterations,
        alpha=0.4,     # High randomization for exploration
        beta0=1.0,
        gamma=0.01     # Low light absorption for global visibility
    )
    
    best_pos_explore, best_fit_explore, history_explore = fa_explore.run()
    
    # Configuration 2: BALANCED
    print("\n" + "-"*70)
    print("Configuration 2: BALANCED")
    print(f"  γ (gamma) = 0.5   [Moderate visibility]")
    print(f"  α (alpha) = 0.2   [Moderate random steps]")
    print(f"  β₀ (beta0) = 1.0  [Standard attractiveness]")
    print("-"*70)
    
    fa_balanced = FireflyAlgorithm(
        objective_func=rastrigin_function,
        dimensions=dimensions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_fireflies=n_fireflies,
        max_iterations=max_iterations,
        alpha=0.2,     # Moderate randomization
        beta0=1.0,
        gamma=0.5      # Moderate light absorption
    )
    
    best_pos_balanced, best_fit_balanced, history_balanced = fa_balanced.run()
    
    # Configuration 3: HIGH EXPLOITATION (suboptimal for Rastrigin)
    print("\n" + "-"*70)
    print("Configuration 3: HIGH EXPLOITATION")
    print(f"  γ (gamma) = 5.0   [High - limited visibility, local search only]")
    print(f"  α (alpha) = 0.05  [Very low - small random steps]")
    print(f"  β₀ (beta0) = 1.0  [Standard attractiveness]")
    print("-"*70)
    
    fa_exploit = FireflyAlgorithm(
        objective_func=rastrigin_function,
        dimensions=dimensions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_fireflies=n_fireflies,
        max_iterations=max_iterations,
        alpha=0.05,    # Low randomization
        beta0=1.0,
        gamma=5.0      # High light absorption
    )
    
    best_pos_exploit, best_fit_exploit, history_exploit = fa_exploit.run()
    
    # Results summary
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"{'Configuration':<25} {'Final Fitness':<20} {'Success'}")
    print("-"*70)
    print(f"{'High Exploration':<25} {best_fit_explore:<20.6f} {'★ BEST' if best_fit_explore < min(best_fit_balanced, best_fit_exploit) else ''}")
    print(f"{'Balanced':<25} {best_fit_balanced:<20.6f}")
    print(f"{'High Exploitation':<25} {best_fit_exploit:<20.6f} {'(Trapped in local optima)' if best_fit_exploit > 50 else ''}")
    print("-"*70)
    print(f"{'Global Optimum':<25} {'0.000000':<20}")
    print("="*70)
    
    print("\n💡 INSIGHT:")
    print("On this COMPLEX, MULTIMODAL problem, HIGH EXPLORATION wins!")
    print("Low γ ensures all fireflies can 'see' the global best, preventing fragmentation.")
    print("High α allows fireflies to escape local optima through large random jumps.")
    print("High γ causes premature convergence - fireflies get trapped in local optima!\n")
    
    return {
        'explore': (history_explore, best_fit_explore),
        'balanced': (history_balanced, best_fit_balanced),
        'exploit': (history_exploit, best_fit_exploit),
        'fa_explore': fa_explore,
        'fa_balanced': fa_balanced,
        'fa_exploit': fa_exploit
    }


def plot_convergence_comparison(results):
    """Plot convergence curves comparing the three configurations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Regular scale
    ax1.plot(results['explore'][0], 'b-', linewidth=2, label='High Exploration (γ=0.01, α=0.4)')
    ax1.plot(results['balanced'][0], 'g-', linewidth=2, label='Balanced (γ=0.5, α=0.2)')
    ax1.plot(results['exploit'][0], 'r-', linewidth=2, label='High Exploitation (γ=5.0, α=0.05)')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Convergence on Rastrigin Function (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Global Optimum')
    
    # Plot 2: Log scale
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    history_explore_log = [max(x, epsilon) for x in results['explore'][0]]
    history_balanced_log = [max(x, epsilon) for x in results['balanced'][0]]
    history_exploit_log = [max(x, epsilon) for x in results['exploit'][0]]
    
    ax2.semilogy(history_explore_log, 'b-', linewidth=2, label='High Exploration (γ=0.01, α=0.4)')
    ax2.semilogy(history_balanced_log, 'g-', linewidth=2, label='Balanced (γ=0.5, α=0.2)')
    ax2.semilogy(history_exploit_log, 'r-', linewidth=2, label='High Exploitation (γ=5.0, α=0.05)')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best Fitness (Log Scale)', fontsize=12)
    ax2.set_title('Convergence on Rastrigin Function (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add annotation box
    textstr = 'Rastrigin Function Properties:\n' \
              '• Highly multimodal (~10^d local minima)\n' \
              '• Periodic cosine modulation\n' \
              '• Ideal for exploration testing\n' \
              '• Global minimum: f(0,...,0) = 0\n' \
              '• Premature convergence = major risk'
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('fa_exploration_convergence.png', dpi=150, bbox_inches='tight')
    print("\n✓ Convergence plot saved as 'fa_exploration_convergence.png'")
    plt.show()


def plot_rastrigin_3d_surface():
    """Plot 3D surface of Rastrigin function to visualize its complexity"""
    print("\nGenerating 3D surface visualization of Rastrigin function...")
    
    fig = plt.figure(figsize=(14, 6))
    
    # Create meshgrid
    x = np.linspace(-5.12, 5.12, 200)
    y = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin_function(np.array([X[i, j], Y[i, j]]))
    
    # Plot 1: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_xlabel('x₁', fontsize=10)
    ax1.set_ylabel('x₂', fontsize=10)
    ax1.set_zlabel('f(x)', fontsize=10)
    ax1.set_title('Rastrigin Function 3D Surface\n(Many Local Minima)', fontsize=12, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Mark global optimum
    ax1.scatter([0], [0], [0], color='red', s=100, marker='*', 
                label='Global Optimum', zorder=5)
    ax1.legend()
    
    # Plot 2: 2D contour
    ax2 = fig.add_subplot(122)
    levels = np.linspace(0, 80, 30)
    contourf = ax2.contourf(X, Y, Z, levels=levels, cmap='viridis')
    contour = ax2.contour(X, Y, Z, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
    ax2.plot(0, 0, 'r*', markersize=20, label='Global Optimum')
    
    ax2.set_xlabel('x₁', fontsize=10)
    ax2.set_ylabel('x₂', fontsize=10)
    ax2.set_title('Rastrigin Function 2D Contour\n(Periodic Structure)', fontsize=12, fontweight='bold')
    ax2.legend()
    fig.colorbar(contourf, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('rastrigin_landscape.png', dpi=150, bbox_inches='tight')
    print("✓ Rastrigin landscape saved as 'rastrigin_landscape.png'")
    plt.show()


def plot_2d_convergence_animation(results):
    """
    Create an animation showing FA convergence on 2D Rastrigin function.
    Compares High Exploration vs High Exploitation side by side.
    """
    print("\nGenerating 2D convergence animation (this may take a moment)...")
    
    # Re-run in 2D for visualization
    dimensions = 2
    lower_bound = -5.12
    upper_bound = 5.12
    n_fireflies = 30
    max_iterations = 100
    
    # High Exploration config
    fa_explore_2d = FireflyAlgorithm(
        objective_func=rastrigin_function,
        dimensions=dimensions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_fireflies=n_fireflies,
        max_iterations=max_iterations,
        alpha=0.4,
        beta0=1.0,
        gamma=0.01
    )
    fa_explore_2d.run()
    
    # High Exploitation config
    fa_exploit_2d = FireflyAlgorithm(
        objective_func=rastrigin_function,
        dimensions=dimensions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_fireflies=n_fireflies,
        max_iterations=max_iterations,
        alpha=0.05,
        beta0=1.0,
        gamma=5.0
    )
    fa_exploit_2d.run()
    
    # Create meshgrid for contour plot
    x = np.linspace(lower_bound, upper_bound, 150)
    y = np.linspace(lower_bound, upper_bound, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin_function(np.array([X[i, j], Y[i, j]]))
    
    # Setup figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot contours for both subplots
    for ax in [ax1, ax2]:
        levels = np.linspace(0, 80, 25)
        contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        contour = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
        ax.plot(0, 0, 'r*', markersize=20, label='Global Optimum', zorder=5)
        ax.set_xlim([lower_bound, upper_bound])
        ax.set_ylim([lower_bound, upper_bound])
        ax.set_xlabel('x₁', fontsize=11)
        ax.set_ylabel('x₂', fontsize=11)
        ax.grid(True, alpha=0.2)
    
    ax1.set_title('High Exploration (γ=0.01, α=0.4)\nGlobal Search', fontsize=12, fontweight='bold')
    ax2.set_title('High Exploitation (γ=5.0, α=0.05)\nLocal Search', fontsize=12, fontweight='bold')
    
    # Initialize scatter plots
    positions_explore = fa_explore_2d.position_history
    positions_exploit = fa_exploit_2d.position_history
    
    scatter1 = ax1.scatter([], [], c='cyan', s=100, alpha=0.7, edgecolors='blue', linewidth=1)
    scatter2 = ax2.scatter([], [], c='yellow', s=100, alpha=0.7, edgecolors='red', linewidth=1)
    
    best_scatter1 = ax1.scatter([], [], c='lime', marker='*', s=400, edgecolors='darkgreen', linewidth=2, zorder=4)
    best_scatter2 = ax2.scatter([], [], c='orange', marker='*', s=400, edgecolors='darkred', linewidth=2, zorder=4)
    
    # Text annotations
    text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9), fontsize=9)
    text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=9)
    
    def animate(frame):
        if frame < len(positions_explore):
            # Update exploration plot
            pos1 = positions_explore[frame]
            scatter1.set_offsets(pos1)
            best_idx1 = np.argmin(fa_explore_2d.intensity_history[frame])
            best_pos1 = pos1[best_idx1]
            best_scatter1.set_offsets([best_pos1])
            best_fit1 = fa_explore_2d.intensity_history[frame][best_idx1]
            
            # Calculate swarm diversity (avg distance between fireflies)
            diversity1 = np.mean([np.linalg.norm(pos1[i] - pos1[j]) 
                                 for i in range(len(pos1)) for j in range(i+1, len(pos1))])
            
            text1.set_text(f'Iteration: {frame}\nBest Fitness: {best_fit1:.4f}\n' +
                          f'Distance from Global: {np.linalg.norm(best_pos1):.3f}\n' +
                          f'Swarm Diversity: {diversity1:.3f}')
        
        if frame < len(positions_exploit):
            # Update exploitation plot
            pos2 = positions_exploit[frame]
            scatter2.set_offsets(pos2)
            best_idx2 = np.argmin(fa_exploit_2d.intensity_history[frame])
            best_pos2 = pos2[best_idx2]
            best_scatter2.set_offsets([best_pos2])
            best_fit2 = fa_exploit_2d.intensity_history[frame][best_idx2]
            
            # Calculate swarm diversity
            diversity2 = np.mean([np.linalg.norm(pos2[i] - pos2[j]) 
                                 for i in range(len(pos2)) for j in range(i+1, len(pos2))])
            
            text2.set_text(f'Iteration: {frame}\nBest Fitness: {best_fit2:.4f}\n' +
                          f'Distance from Global: {np.linalg.norm(best_pos2):.3f}\n' +
                          f'Swarm Diversity: {diversity2:.3f}')
        
        return scatter1, scatter2, best_scatter1, best_scatter2, text1, text2
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=max_iterations, interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    
    # Save animation
    try:
        anim.save('fa_exploration_animation.gif', writer='pillow', fps=10, dpi=100)
        print("✓ Animation saved as 'fa_exploration_animation.gif'")
    except Exception as e:
        print(f"✗ Could not save animation: {e}")
        print("  Install pillow: pip install pillow")
    
    plt.show()


def compare_multiple_runs():
    """
    Run multiple trials to demonstrate consistency of exploration vs exploitation
    on multimodal problems.
    """
    print("\nRunning multiple trials for statistical comparison...")
    
    dimensions = 10
    lower_bound = -5.12
    upper_bound = 5.12
    n_fireflies = 40
    max_iterations = 150
    n_trials = 10
    
    results_explore = []
    results_exploit = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...", end='\r')
        
        # High Exploration
        fa_explore = FireflyAlgorithm(
            objective_func=rastrigin_function,
            dimensions=dimensions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.4,
            beta0=1.0,
            gamma=0.01
        )
        _, best_fit_explore, _ = fa_explore.run()
        results_explore.append(best_fit_explore)
        
        # High Exploitation
        fa_exploit = FireflyAlgorithm(
            objective_func=rastrigin_function,
            dimensions=dimensions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.05,
            beta0=1.0,
            gamma=5.0
        )
        _, best_fit_exploit, _ = fa_exploit.run()
        results_exploit.append(best_fit_exploit)
    
    print("\n\n" + "="*70)
    print("STATISTICAL COMPARISON ({} trials)".format(n_trials))
    print("="*70)
    print(f"{'Metric':<30} {'High Exploration':<20} {'High Exploitation'}")
    print("-"*70)
    print(f"{'Mean Best Fitness':<30} {np.mean(results_explore):<20.4f} {np.mean(results_exploit):.4f}")
    print(f"{'Std Dev':<30} {np.std(results_explore):<20.4f} {np.std(results_exploit):.4f}")
    print(f"{'Best Result':<30} {np.min(results_explore):<20.4f} {np.min(results_exploit):.4f}")
    print(f"{'Worst Result':<30} {np.max(results_explore):<20.4f} {np.max(results_exploit):.4f}")
    print(f"{'Success Rate (< 10)':<30} {sum(x < 10 for x in results_explore)/n_trials*100:<19.1f}% {sum(x < 10 for x in results_exploit)/n_trials*100:.1f}%")
    print("="*70)
    
    # Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_data = [results_explore, results_exploit]
    bp = ax.boxplot(box_data, labels=['High Exploration\n(γ=0.01, α=0.4)', 'High Exploitation\n(γ=5.0, α=0.05)'],
                    patch_artist=True, widths=0.6)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Global Optimum (0)')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Success Threshold (10)')
    
    ax.set_ylabel('Final Best Fitness', fontsize=12)
    ax.set_title(f'Performance Distribution on Rastrigin Function\n({n_trials} Independent Runs)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fa_exploration_statistics.png', dpi=150, bbox_inches='tight')
    print("\n✓ Statistical comparison plot saved as 'fa_exploration_statistics.png'")
    plt.show()


def main():
    """Run the exploration demonstration"""
    print("\n" + "="*70)
    print(" "*15 + "FA EXPLORATION DEMO")
    print(" "*8 + "Complex Multimodal Problem (Rastrigin Function)")
    print("="*70 + "\n")
    
    # Show the challenging landscape
    plot_rastrigin_3d_surface()
    
    # Run experiments
    results = run_exploration_experiment()
    
    # Plot convergence comparison
    plot_convergence_comparison(results)
    
    # Create 2D animation
    plot_2d_convergence_animation(results)
    
    # Statistical comparison
    compare_multiple_runs()
    
    print("\n" + "="*70)
    print("EXPLORATION DEMO COMPLETE")
    print("="*70)
    print("\n📝 KEY TAKEAWAYS:")
    print("1. Low γ (gamma) enables global visibility → all fireflies see the best")
    print("2. High α (alpha) allows large jumps → escape from local optima")
    print("3. On MULTIMODAL problems, exploration-focused parameters find global optimum")
    print("4. High γ causes premature convergence → swarm fragments & gets trapped")
    print("5. The challenge: balance exploration to escape traps vs exploitation to converge")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
