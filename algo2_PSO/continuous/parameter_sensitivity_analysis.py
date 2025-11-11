"""
PSO Parameter Sensitivity Analysis
===================================
This file performs comprehensive sensitivity analysis of PSO parameters:
- Inertia Weight (w): Controls momentum and exploration/exploitation balance
- Cognitive Coefficient (c1): Controls individual particle's trust in its own experience
- Social Coefficient (c2): Controls particle's trust in swarm's collective knowledge

The analysis tests various parameter combinations on both simple (Sphere) and 
complex (Rastrigin) problems to demonstrate:
1. How each parameter affects convergence speed
2. How parameter balance affects solution quality
3. The importance of problem-specific parameter tuning
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algo2_PSO.continuous.pso import PSO
from utils.Continuous_functions import rastrigin


# --- Objective Functions ---
def sphere(X):
    """Sphere function - simple unimodal problem."""
    return np.sum(X**2, axis=0)


def run_parameter_sweep(obj_func, func_name, bounds_low, bounds_high, 
                        w_values, c1_values, c2_values, 
                        n_particles=20, max_iterations=100, n_trials=5):
    """
    Run PSO with different parameter combinations and collect results.
    
    Returns:
        Dictionary mapping (w, c1, c2) tuples to performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Parameter Sensitivity Analysis: {func_name}")
    print(f"{'='*70}")
    
    results = {}
    total_combinations = len(w_values) * len(c1_values) * len(c2_values)
    current = 0
    
    for w in w_values:
        for c1 in c1_values:
            for c2 in c2_values:
                current += 1
                print(f"\rTesting combination {current}/{total_combinations}: "
                      f"w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}...", end='')
                
                # Run multiple trials
                trial_results = []
                for trial in range(n_trials):
                    pso = PSO(
                        obj_func=obj_func,
                        n_particles=n_particles,
                        n_dims=2,
                        bounds_low=bounds_low,
                        bounds_high=bounds_high,
                        w=w,
                        c1=c1,
                        c2=c2
                    )
                    
                    pos, fit, fit_hist, _, _ = pso.optimize(max_iterations)
                    trial_results.append({
                        'final_fitness': fit,
                        'convergence_history': fit_hist,
                        'final_position': pos
                    })
                
                # Compute statistics
                final_fitnesses = [r['final_fitness'] for r in trial_results]
                avg_fitness = np.mean(final_fitnesses)
                std_fitness = np.std(final_fitnesses)
                best_fitness = np.min(final_fitnesses)
                
                # Convergence speed (iteration to reach threshold)
                threshold = 1.0 if func_name == "Rastrigin" else 0.1
                convergence_iterations = []
                for r in trial_results:
                    conv_iter = next((i for i, f in enumerate(r['convergence_history']) 
                                    if f < threshold), max_iterations)
                    convergence_iterations.append(conv_iter)
                avg_convergence_speed = np.mean(convergence_iterations)
                
                results[(w, c1, c2)] = {
                    'avg_fitness': avg_fitness,
                    'std_fitness': std_fitness,
                    'best_fitness': best_fitness,
                    'avg_convergence_speed': avg_convergence_speed,
                    'convergence_history': trial_results[0]['convergence_history']  # Use first trial
                }
    
    print(f"\r{'✓ Completed!':<70}")
    return results


def analyze_inertia_weight(obj_func, func_name, bounds_low, bounds_high):
    """Analyze the effect of inertia weight (w) while keeping c1=c2=1.5."""
    print(f"\n{'─'*70}")
    print("Analysis 1: Inertia Weight (w) Sensitivity")
    print(f"{'─'*70}")
    print("Fixed: c1=1.5, c2=1.5 (balanced)")
    print("Variable: w ∈ [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]")
    
    w_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c1_values = [1.5]
    c2_values = [1.5]
    
    results = run_parameter_sweep(
        obj_func, func_name, bounds_low, bounds_high,
        w_values, c1_values, c2_values
    )
    
    return results, w_values


def analyze_cognitive_coefficient(obj_func, func_name, bounds_low, bounds_high):
    """Analyze the effect of cognitive coefficient (c1) while keeping w=0.7, c2=1.5."""
    print(f"\n{'─'*70}")
    print("Analysis 2: Cognitive Coefficient (c1) Sensitivity")
    print(f"{'─'*70}")
    print("Fixed: w=0.7, c2=1.5")
    print("Variable: c1 ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]")
    
    w_values = [0.7]
    c1_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    c2_values = [1.5]
    
    results = run_parameter_sweep(
        obj_func, func_name, bounds_low, bounds_high,
        w_values, c1_values, c2_values
    )
    
    return results, c1_values


def analyze_social_coefficient(obj_func, func_name, bounds_low, bounds_high):
    """Analyze the effect of social coefficient (c2) while keeping w=0.7, c1=1.5."""
    print(f"\n{'─'*70}")
    print("Analysis 3: Social Coefficient (c2) Sensitivity")
    print(f"{'─'*70}")
    print("Fixed: w=0.7, c1=1.5")
    print("Variable: c2 ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]")
    
    w_values = [0.7]
    c1_values = [1.5]
    c2_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = run_parameter_sweep(
        obj_func, func_name, bounds_low, bounds_high,
        w_values, c1_values, c2_values
    )
    
    return results, c2_values


def analyze_balance_c1_c2(obj_func, func_name, bounds_low, bounds_high):
    """Analyze the balance between c1 and c2 while keeping w=0.7."""
    print(f"\n{'─'*70}")
    print("Analysis 4: c1/c2 Balance (Exploration vs Exploitation)")
    print(f"{'─'*70}")
    print("Fixed: w=0.7, c1+c2=3.0")
    print("Variable: c1/c2 ratio")
    
    w_values = [0.7]
    # Keep sum constant, vary ratio
    pairs = [
        (0.5, 2.5),   # c2 >> c1: High exploitation
        (1.0, 2.0),   # c2 > c1: Moderate exploitation
        (1.5, 1.5),   # c2 = c1: Balanced
        (2.0, 1.0),   # c1 > c2: Moderate exploration
        (2.5, 0.5),   # c1 >> c2: High exploration
    ]
    
    c1_values = [p[0] for p in pairs]
    c2_values = [p[1] for p in pairs]
    
    results = {}
    for c1, c2 in pairs:
        result = run_parameter_sweep(
            obj_func, func_name, bounds_low, bounds_high,
            w_values, [c1], [c2], n_trials=5
        )
        results[(0.7, c1, c2)] = result[(0.7, c1, c2)]
    
    return results, pairs


def create_comprehensive_visualization(
    w_results_sphere, w_values_sphere,
    c1_results_sphere, c1_values_sphere,
    c2_results_sphere, c2_values_sphere,
    balance_results_sphere, balance_pairs_sphere,
    w_results_rastrigin, w_values_rastrigin,
    c1_results_rastrigin, c1_values_rastrigin,
    c2_results_rastrigin, c2_values_rastrigin,
    balance_results_rastrigin, balance_pairs_rastrigin
):
    """Create comprehensive visualization of all sensitivity analyses."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Color scheme
    color_sphere = 'steelblue'
    color_rastrigin = 'darkred'
    
    # ========== ROW 1: Inertia Weight (w) Analysis ==========
    
    # Sphere - w sensitivity
    ax1 = fig.add_subplot(gs[0, 0])
    w_fitness_sphere = [w_results_sphere[(w, 1.5, 1.5)]['avg_fitness'] for w in w_values_sphere]
    ax1.plot(w_values_sphere, w_fitness_sphere, 'o-', color=color_sphere, linewidth=2, markersize=8)
    ax1.set_xlabel('Inertia Weight (w)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax1.set_title('Sphere: w Sensitivity\n(Simple Unimodal)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='Optimal w≈0.4')
    ax1.legend(fontsize=8)
    
    # Rastrigin - w sensitivity
    ax2 = fig.add_subplot(gs[0, 1])
    w_fitness_rastrigin = [w_results_rastrigin[(w, 1.5, 1.5)]['avg_fitness'] for w in w_values_rastrigin]
    ax2.plot(w_values_rastrigin, w_fitness_rastrigin, 'o-', color=color_rastrigin, linewidth=2, markersize=8)
    ax2.set_xlabel('Inertia Weight (w)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax2.set_title('Rastrigin: w Sensitivity\n(Complex Multimodal)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Optimal w≈0.7-0.9')
    ax2.axvline(x=0.9, color='green', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8)
    
    # Convergence comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(w_results_sphere[(0.4, 1.5, 1.5)]['convergence_history'], 
                 color=color_sphere, linewidth=2, label='Sphere (w=0.4)')
    ax3.semilogy(w_results_sphere[(0.9, 1.5, 1.5)]['convergence_history'], 
                 color=color_sphere, linewidth=2, linestyle='--', label='Sphere (w=0.9)')
    ax3.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Best Fitness (log)', fontsize=10, fontweight='bold')
    ax3.set_title('Sphere: Convergence\n(Low vs High w)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.semilogy(w_results_rastrigin[(0.4, 1.5, 1.5)]['convergence_history'], 
                 color=color_rastrigin, linewidth=2, label='Rastrigin (w=0.4)')
    ax4.semilogy(w_results_rastrigin[(0.9, 1.5, 1.5)]['convergence_history'], 
                 color=color_rastrigin, linewidth=2, linestyle='--', label='Rastrigin (w=0.9)')
    ax4.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Best Fitness (log)', fontsize=10, fontweight='bold')
    ax4.set_title('Rastrigin: Convergence\n(Low vs High w)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # ========== ROW 2: Cognitive (c1) and Social (c2) Coefficients ==========
    
    # Sphere - c1 sensitivity
    ax5 = fig.add_subplot(gs[1, 0])
    c1_fitness_sphere = [c1_results_sphere[(0.7, c1, 1.5)]['avg_fitness'] for c1 in c1_values_sphere]
    ax5.plot(c1_values_sphere, c1_fitness_sphere, 's-', color=color_sphere, linewidth=2, markersize=8)
    ax5.set_xlabel('Cognitive Coeff (c1)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax5.set_title('Sphere: c1 Sensitivity', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Rastrigin - c1 sensitivity
    ax6 = fig.add_subplot(gs[1, 1])
    c1_fitness_rastrigin = [c1_results_rastrigin[(0.7, c1, 1.5)]['avg_fitness'] for c1 in c1_values_rastrigin]
    ax6.plot(c1_values_rastrigin, c1_fitness_rastrigin, 's-', color=color_rastrigin, linewidth=2, markersize=8)
    ax6.set_xlabel('Cognitive Coeff (c1)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax6.set_title('Rastrigin: c1 Sensitivity', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Higher c1 better')
    ax6.legend(fontsize=8)
    
    # Sphere - c2 sensitivity
    ax7 = fig.add_subplot(gs[1, 2])
    c2_fitness_sphere = [c2_results_sphere[(0.7, 1.5, c2)]['avg_fitness'] for c2 in c2_values_sphere]
    ax7.plot(c2_values_sphere, c2_fitness_sphere, '^-', color=color_sphere, linewidth=2, markersize=8)
    ax7.set_xlabel('Social Coeff (c2)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax7.set_title('Sphere: c2 Sensitivity', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axvline(x=2.5, color='green', linestyle='--', alpha=0.5, label='Higher c2 better')
    ax7.legend(fontsize=8)
    
    # Rastrigin - c2 sensitivity
    ax8 = fig.add_subplot(gs[1, 3])
    c2_fitness_rastrigin = [c2_results_rastrigin[(0.7, 1.5, c2)]['avg_fitness'] for c2 in c2_values_rastrigin]
    ax8.plot(c2_values_rastrigin, c2_fitness_rastrigin, '^-', color=color_rastrigin, linewidth=2, markersize=8)
    ax8.set_xlabel('Social Coeff (c2)', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax8.set_title('Rastrigin: c2 Sensitivity', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # ========== ROW 3: Balance Analysis (c1 vs c2) ==========
    
    # Sphere - balance
    ax9 = fig.add_subplot(gs[2, 0:2])
    labels_sphere = [f"c1={c1:.1f}\nc2={c2:.1f}" for c1, c2 in balance_pairs_sphere]
    balance_fitness_sphere = [balance_results_sphere[(0.7, c1, c2)]['avg_fitness'] 
                              for c1, c2 in balance_pairs_sphere]
    colors_sphere = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bars = ax9.bar(range(len(labels_sphere)), balance_fitness_sphere, color=colors_sphere, 
                   edgecolor='black', linewidth=1.5)
    ax9.set_xticks(range(len(labels_sphere)))
    ax9.set_xticklabels(labels_sphere, fontsize=9)
    ax9.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax9.set_title('Sphere: Exploration-Exploitation Balance\n(c1+c2=3.0, w=0.7)', 
                  fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax9.text(0, max(balance_fitness_sphere)*0.9, 'High\nExploitation', 
             ha='center', fontsize=9, fontweight='bold', color='darkred')
    ax9.text(2, max(balance_fitness_sphere)*0.9, 'Balanced', 
             ha='center', fontsize=9, fontweight='bold', color='black')
    ax9.text(4, max(balance_fitness_sphere)*0.9, 'High\nExploration', 
             ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    
    # Rastrigin - balance
    ax10 = fig.add_subplot(gs[2, 2:4])
    labels_rastrigin = [f"c1={c1:.1f}\nc2={c2:.1f}" for c1, c2 in balance_pairs_rastrigin]
    balance_fitness_rastrigin = [balance_results_rastrigin[(0.7, c1, c2)]['avg_fitness'] 
                                 for c1, c2 in balance_pairs_rastrigin]
    colors_rastrigin = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bars = ax10.bar(range(len(labels_rastrigin)), balance_fitness_rastrigin, 
                    color=colors_rastrigin, edgecolor='black', linewidth=1.5)
    ax10.set_xticks(range(len(labels_rastrigin)))
    ax10.set_xticklabels(labels_rastrigin, fontsize=9)
    ax10.set_ylabel('Final Fitness', fontsize=10, fontweight='bold')
    ax10.set_title('Rastrigin: Exploration-Exploitation Balance\n(c1+c2=3.0, w=0.7)', 
                   fontsize=11, fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax10.text(0, max(balance_fitness_rastrigin)*0.9, 'High\nExploitation', 
              ha='center', fontsize=9, fontweight='bold', color='darkred')
    ax10.text(2, max(balance_fitness_rastrigin)*0.9, 'Balanced', 
              ha='center', fontsize=9, fontweight='bold', color='black')
    ax10.text(4, max(balance_fitness_rastrigin)*0.9, 'High\nExploration', 
              ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    
    plt.suptitle('PSO Parameter Sensitivity Analysis: Comprehensive Study', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = Path(__file__).parent / "parameter_sensitivity_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    
    return fig


def print_recommendations():
    """Print parameter tuning recommendations."""
    print("\n" + "="*70)
    print("PARAMETER TUNING RECOMMENDATIONS")
    print("="*70)
    
    print("\n┌─ FOR SIMPLE UNIMODAL PROBLEMS (e.g., Sphere) ─┐")
    print("│                                                 │")
    print("│  GOAL: Fast convergence to single optimum      │")
    print("│                                                 │")
    print("│  • Inertia Weight (w): 0.4 - 0.6 [LOW]        │")
    print("│    → Quick braking, focus on current region    │")
    print("│                                                 │")
    print("│  • Cognitive (c1): 1.0 - 1.5 [MODERATE]       │")
    print("│    → Some individual search                     │")
    print("│                                                 │")
    print("│  • Social (c2): 2.0 - 2.5 [HIGH]              │")
    print("│    → Strong consensus, fast convergence        │")
    print("│                                                 │")
    print("│  • Balance: c2 > c1 (EXPLOITATION)             │")
    print("│                                                 │")
    print("└─────────────────────────────────────────────────┘")
    
    print("\n┌─ FOR COMPLEX MULTIMODAL PROBLEMS (e.g., Rastrigin) ─┐")
    print("│                                                       │")
    print("│  GOAL: Avoid local minima, find global optimum       │")
    print("│                                                       │")
    print("│  • Inertia Weight (w): 0.7 - 0.9 [HIGH]             │")
    print("│    → Strong momentum to escape local minima          │")
    print("│                                                       │")
    print("│  • Cognitive (c1): 1.5 - 2.5 [HIGH]                 │")
    print("│    → Individual exploration, diversity               │")
    print("│                                                       │")
    print("│  • Social (c2): 1.0 - 1.5 [MODERATE]                │")
    print("│    → Avoid premature convergence                     │")
    print("│                                                       │")
    print("│  • Balance: c1 > c2 (EXPLORATION)                    │")
    print("│                                                       │")
    print("└───────────────────────────────────────────────────────┘")
    
    print("\n┌─ GENERAL GUIDELINES ─┐")
    print("│                       │")
    print("│  • Start with balanced settings: w=0.729, c1=1.49, c2=1.49  │")
    print("│                                                               │")
    print("│  • If converging too fast (premature):                      │")
    print("│    → Increase w (more exploration)                           │")
    print("│    → Increase c1 (more individual search)                    │")
    print("│    → Decrease c2 (less consensus)                            │")
    print("│                                                               │")
    print("│  • If converging too slow:                                   │")
    print("│    → Decrease w (more exploitation)                          │")
    print("│    → Increase c2 (more consensus)                            │")
    print("│    → Decrease c1 (less individual search)                    │")
    print("│                                                               │")
    print("│  • CRITICAL: c2 >> c1 can cause premature convergence!       │")
    print("│                                                               │")
    print("└───────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PSO PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nThis analysis will test PSO performance with various parameter")
    print("combinations on both simple (Sphere) and complex (Rastrigin) problems.")
    print("\nNote: This may take several minutes to complete...")
    
    # ========== SPHERE FUNCTION (Simple) ==========
    print("\n" + "#"*70)
    print("# PART 1: SPHERE FUNCTION (Simple Unimodal Problem)")
    print("#"*70)
    
    w_results_sphere, w_values_sphere = analyze_inertia_weight(
        sphere, "Sphere", -10.0, 10.0
    )
    
    c1_results_sphere, c1_values_sphere = analyze_cognitive_coefficient(
        sphere, "Sphere", -10.0, 10.0
    )
    
    c2_results_sphere, c2_values_sphere = analyze_social_coefficient(
        sphere, "Sphere", -10.0, 10.0
    )
    
    balance_results_sphere, balance_pairs_sphere = analyze_balance_c1_c2(
        sphere, "Sphere", -10.0, 10.0
    )
    
    # ========== RASTRIGIN FUNCTION (Complex) ==========
    print("\n" + "#"*70)
    print("# PART 2: RASTRIGIN FUNCTION (Complex Multimodal Problem)")
    print("#"*70)
    
    w_results_rastrigin, w_values_rastrigin = analyze_inertia_weight(
        rastrigin, "Rastrigin", -5.12, 5.12
    )
    
    c1_results_rastrigin, c1_values_rastrigin = analyze_cognitive_coefficient(
        rastrigin, "Rastrigin", -5.12, 5.12
    )
    
    c2_results_rastrigin, c2_values_rastrigin = analyze_social_coefficient(
        rastrigin, "Rastrigin", -5.12, 5.12
    )
    
    balance_results_rastrigin, balance_pairs_rastrigin = analyze_balance_c1_c2(
        rastrigin, "Rastrigin", -5.12, 5.12
    )
    
    # ========== CREATE VISUALIZATION ==========
    print("\n" + "="*70)
    print("Creating comprehensive visualization...")
    print("="*70)
    
    create_comprehensive_visualization(
        w_results_sphere, w_values_sphere,
        c1_results_sphere, c1_values_sphere,
        c2_results_sphere, c2_values_sphere,
        balance_results_sphere, balance_pairs_sphere,
        w_results_rastrigin, w_values_rastrigin,
        c1_results_rastrigin, c1_values_rastrigin,
        c2_results_rastrigin, c2_values_rastrigin,
        balance_results_rastrigin, balance_pairs_rastrigin
    )
    
    # ========== PRINT RECOMMENDATIONS ==========
    print_recommendations()
    
    # Show plot
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis completed!")
    print("Check 'parameter_sensitivity_analysis.png' for comprehensive results.")
    print("="*70)
