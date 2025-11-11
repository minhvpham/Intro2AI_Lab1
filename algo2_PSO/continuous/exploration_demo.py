"""
PSO Exploration Demo: Complex Multimodal Problem
=================================================
This file demonstrates how HIGH exploration (high w, high c1 > c2) allows PSO 
to avoid premature convergence on a complex multimodal problem (Rastrigin function).

Core Concept:
- High w (inertia weight): Particles maintain momentum, can overshoot local minima
- High c1 (cognitive coefficient): Strong individual exploration, diversity
- Low c2 (social coefficient): Weak swarm consensus, avoid premature convergence

Expected Behavior:
- Slower but more thorough search
- Better at escaping local minima
- Essential for complex multimodal problems
- May overshoot on simple problems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algo2_PSO.continuous.pso import PSO, rastrigin_plot
from utils.Continuous_functions import rastrigin


def run_exploration_demo():
    """
    Run PSO on RASTRIGIN function (complex problem) with HIGH EXPLORATION settings.
    """
    print("=" * 70)
    print("EXPLORATION DEMO: Complex Multimodal Problem (Rastrigin Function)")
    print("=" * 70)
    print("\nScenario: 2D Rastrigin function - Complex problem with many local minima")
    print("Strategy: High Exploration (Avoid premature convergence, find global optimum)")
    print("\nParameter Settings:")
    
    # Problem parameters
    dims = 2
    bounds_low = -5.12
    bounds_high = 5.12
    n_particles = 30  # More particles for complex problems
    max_iterations = 150  # More iterations needed
    
    # EXPLORATION SETTINGS
    w = 0.9      # HIGH: Strong momentum to overshoot local minima
    c1 = 2.0     # HIGH: Strong individual exploration
    c2 = 1.0     # LOW: Weak swarm consensus (maintain diversity)
    
    print(f"  • Inertia Weight (w) = {w:.2f}  [HIGH - Strong momentum]")
    print(f"  • Cognitive Coeff (c1) = {c1:.2f}  [HIGH - Individual exploration]")
    print(f"  • Social Coeff (c2) = {c2:.2f}  [LOW - Weak consensus]")
    print(f"  • Particles: {n_particles}")
    print(f"  • Max Iterations: {max_iterations}")
    print(f"\n{'Interpretation:':>20} c1 > c2 and high w → Thorough exploration!")
    
    # Run PSO with exploration settings
    print("\n" + "-" * 70)
    print("Running PSO with EXPLORATION settings...")
    print("-" * 70)
    
    pso_explore = PSO(
        obj_func=rastrigin,
        n_particles=n_particles,
        n_dims=dims,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        w=w,
        c1=c1,
        c2=c2
    )
    
    pos, fit, fit_hist, pos_hist, gbest_pos_hist = pso_explore.optimize(max_iterations)
    
    print(f"\nBest Fitness: {fit:.6f}")
    print(f"Best Position: [{pos[0]:.6f}, {pos[1]:.6f}]")
    print(f"True Optimum: [0.000000, 0.000000] (fitness = 0.0)")
    print(f"Distance to optimum: {np.sqrt(np.sum(pos**2)):.6f}")
    
    # For comparison, run with exploitation settings (DANGER!)
    print("\n" + "=" * 70)
    print("COMPARISON: Running with EXPLOITATION settings on same problem")
    print("=" * 70)
    print("\nParameter Settings:")
    
    # EXPLOITATION SETTINGS (BAD for complex problems)
    w_exploit = 0.4      # LOW: Quick braking
    c1_exploit = 1.0     # LOW: Less individual search
    c2_exploit = 2.5     # HIGH: Strong consensus (premature convergence risk!)
    
    print(f"  • Inertia Weight (w) = {w_exploit:.2f}  [LOW - Quick braking]")
    print(f"  • Cognitive Coeff (c1) = {c1_exploit:.2f}  [LOW - Less diversity]")
    print(f"  • Social Coeff (c2) = {c2_exploit:.2f}  [HIGH - Strong consensus]")
    print(f"\n{'WARNING:':>20} c2 >> c1 → Risk of PREMATURE CONVERGENCE!")
    
    print("\n" + "-" * 70)
    print("Running PSO with EXPLOITATION settings...")
    print("-" * 70)
    
    pso_exploit = PSO(
        obj_func=rastrigin,
        n_particles=n_particles,
        n_dims=dims,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        w=w_exploit,
        c1=c1_exploit,
        c2=c2_exploit
    )
    
    pos_exploit, fit_exploit, fit_hist_exploit, pos_hist_exploit, gbest_pos_hist_exploit = pso_exploit.optimize(max_iterations)
    
    print(f"\nBest Fitness: {fit_exploit:.6f}")
    print(f"Best Position: [{pos_exploit[0]:.6f}, {pos_exploit[1]:.6f}]")
    print(f"Distance to optimum: {np.sqrt(np.sum(pos_exploit**2)):.6f}")
    
    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Final Fitness':<20} {'Quality':<20}")
    print("-" * 70)
    print(f"{'EXPLORATION':<20} {fit:.6f}{'':<13} {'⭐ Better' if fit < fit_exploit else ''}")
    print(f"{'EXPLOITATION':<20} {fit_exploit:.6f}{'':<13} {'⭐ Better' if fit_exploit < fit else '⚠ Trapped?'}")
    print("-" * 70)
    
    if fit < fit_exploit:
        improvement = ((fit_exploit - fit) / fit_exploit) * 100
        print(f"\n✓ EXPLORATION achieves {improvement:.1f}% better solution quality!")
        print("  → Successfully avoided local minima")
    else:
        print(f"\n⚠ EXPLOITATION may have gotten lucky or EXPLORATION needs more iterations")
    
    # Calculate diversity (average distance between particles)
    def calculate_diversity(positions):
        """Calculate average pairwise distance between particles."""
        n = len(positions)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total_dist += np.linalg.norm(positions[i] - positions[j])
                count += 1
        return total_dist / count if count > 0 else 0
    
    final_diversity_explore = calculate_diversity(pso_explore.positions)
    final_diversity_exploit = calculate_diversity(pso_exploit.positions)
    
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)
    print(f"Final swarm diversity (average inter-particle distance):")
    print(f"  • Exploration: {final_diversity_explore:.4f}")
    print(f"  • Exploitation: {final_diversity_exploit:.4f}")
    
    if final_diversity_explore > final_diversity_exploit:
        ratio = final_diversity_explore / final_diversity_exploit
        print(f"\n✓ Exploration maintains {ratio:.2f}x more diversity")
        print("  → Reduces risk of premature convergence")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: For Complex Multimodal Problems")
    print("=" * 70)
    print("• Use HIGH inertia (w ≈ 0.7-0.9) to overshoot local minima")
    print("• Use HIGH cognitive coefficient (c1 > c2) for diversity")
    print("• Exploration settings avoid premature convergence")
    print("• Essential for problems with many local optima (like Rastrigin)!")
    print("• Trade-off: Slower convergence but better final solution quality")
    
    return pso_explore, pso_exploit, fit_hist, fit_hist_exploit


def create_visualization(pso_explore, pso_exploit, fit_hist_explore, fit_hist_exploit):
    """Create comprehensive visualization comparing exploration vs exploitation."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    bounds_low, bounds_high = -5.12, 5.12
    
    # --- Row 1: Convergence Curves ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(fit_hist_explore, 'b-', linewidth=2.5, label='Exploration (w=0.9, c1=2.0, c2=1.0)')
    ax1.semilogy(fit_hist_exploit, 'r--', linewidth=2.5, label='Exploitation (w=0.4, c1=1.0, c2=2.5)')
    ax1.axhline(y=1.0, color='g', linestyle=':', linewidth=2, label='Good Solution Threshold')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Fitness (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Comparison: Rastrigin Function (Complex Multimodal)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # --- Row 2: Final Particle Distributions ---
    x = np.linspace(bounds_low, bounds_high, 200)
    y = np.linspace(bounds_low, bounds_high, 200)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_plot(X, Y)
    
    # Exploration final state
    ax2 = fig.add_subplot(gs[1, 0])
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    final_pos_explore = pso_explore.positions
    ax2.scatter(final_pos_explore[:, 0], final_pos_explore[:, 1], 
                c='cyan', marker='o', s=100, alpha=0.8, edgecolors='blue', linewidths=2,
                label='Particles')
    ax2.scatter([pso_explore.gbest_position[0]], [pso_explore.gbest_position[1]], 
                c='red', marker='*', s=400, edgecolors='darkred', linewidths=2,
                label='Best Found', zorder=10)
    ax2.plot(0, 0, 'white', marker='x', markersize=15, markeredgewidth=3,
             label='True Optimum')
    
    ax2.set_xlim([bounds_low, bounds_high])
    ax2.set_ylim([bounds_low, bounds_high])
    ax2.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax2.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax2.set_title('EXPLORATION: Final State (High Diversity)', 
                  fontsize=12, fontweight='bold', color='blue')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Exploitation final state
    ax3 = fig.add_subplot(gs[1, 1])
    contour = ax3.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax3.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    final_pos_exploit = pso_exploit.positions
    ax3.scatter(final_pos_exploit[:, 0], final_pos_exploit[:, 1], 
                c='orange', marker='o', s=100, alpha=0.8, edgecolors='red', linewidths=2,
                label='Particles')
    ax3.scatter([pso_exploit.gbest_position[0]], [pso_exploit.gbest_position[1]], 
                c='red', marker='*', s=400, edgecolors='darkred', linewidths=2,
                label='Best Found', zorder=10)
    ax3.plot(0, 0, 'white', marker='x', markersize=15, markeredgewidth=3,
             label='True Optimum')
    
    ax3.set_xlim([bounds_low, bounds_high])
    ax3.set_ylim([bounds_low, bounds_high])
    ax3.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax3.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax3.set_title('EXPLOITATION: Final State (Low Diversity)', 
                  fontsize=12, fontweight='bold', color='red')
    ax3.legend(loc='upper right', fontsize=9)
    
    # --- Row 3: Diversity Evolution ---
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Calculate diversity over time
    def calculate_diversity_history(position_history):
        diversity = []
        for positions in position_history:
            n = len(positions)
            total_dist = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_dist += np.linalg.norm(positions[i] - positions[j])
                    count += 1
            diversity.append(total_dist / count if count > 0 else 0)
        return diversity
    
    diversity_explore = calculate_diversity_history(pso_explore.position_history)
    diversity_exploit = calculate_diversity_history(pso_exploit.position_history)
    
    ax4.plot(diversity_explore, 'b-', linewidth=2, label='Exploration')
    ax4.plot(diversity_exploit, 'r--', linewidth=2, label='Exploitation')
    ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Swarm Diversity', fontsize=11, fontweight='bold')
    ax4.set_title('Diversity Evolution (Higher = More Exploration)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # --- Row 3: Performance Summary ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = f"""
PERFORMANCE SUMMARY
{'='*40}

EXPLORATION (w=0.9, c1=2.0, c2=1.0):
  • Final Fitness: {pso_explore.gbest_fitness:.6f}
  • Final Diversity: {diversity_explore[-1]:.4f}
  • Strategy: Maintain diversity, avoid 
    premature convergence

EXPLOITATION (w=0.4, c1=1.0, c2=2.5):
  • Final Fitness: {pso_exploit.gbest_fitness:.6f}
  • Final Diversity: {diversity_exploit[-1]:.4f}
  • Strategy: Fast convergence, risk of 
    getting trapped in local minima

KEY OBSERVATIONS:
  ✓ Exploration maintains higher diversity
  ✓ Better for complex multimodal problems
  ✓ Trade-off: slower but more thorough
  
RECOMMENDATION FOR RASTRIGIN:
  Use EXPLORATION settings (high w, c1>c2)
  to avoid the many local minima!
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PSO EXPLORATION vs EXPLOITATION: Complex Multimodal Problem', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = Path(__file__).parent / "exploration_demo_rastrigin.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Run demo
    pso_explore, pso_exploit, fit_hist_explore, fit_hist_exploit = run_exploration_demo()
    
    # Create visualization
    create_visualization(pso_explore, pso_exploit, fit_hist_explore, fit_hist_exploit)
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 70)
    print("Demo completed! Check 'exploration_demo_rastrigin.png' for visualization.")
    print("=" * 70)
