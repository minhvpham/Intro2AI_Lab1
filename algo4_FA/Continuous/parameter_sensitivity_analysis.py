"""
Parameter Sensitivity Analysis: Firefly Algorithm

This comprehensive analysis demonstrates how the three key FA parameters (γ, α, β₀) 
affect the algorithm's performance through systematic experiments and visualizations.

The analysis includes:
1. Individual parameter sensitivity (varying one parameter at a time)
2. Parameter interaction effects (2D heatmaps)
3. Performance across different problem types (unimodal vs multimodal)
4. Convergence behavior analysis
5. Practical parameter tuning guidelines
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path
from itertools import product
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algo4_FA.Continuous.FA import FireflyAlgorithm


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def sphere_function(X):
    """Unimodal function - Single global optimum at origin"""
    return np.sum(X**2)


def rastrigin_function(X):
    """Highly multimodal function - Many local optima"""
    A = 10
    n_dims = len(X)
    return A * n_dims + np.sum(X**2 - A * np.cos(2 * np.pi * X))


def rosenbrock_function(X):
    """Valley-shaped function - Difficult for optimization"""
    return np.sum(100.0 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)


def ackley_function(X):
    """Multimodal function with nearly flat outer region"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(X)
    
    sum1 = np.sum(X**2)
    sum2 = np.sum(np.cos(c * X))
    
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)


# ============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

def analyze_gamma_sensitivity():
    """
    Analyze the effect of γ (light absorption coefficient) on performance.
    
    γ controls visibility:
    - Low γ (0.001-0.1): Global visibility, swarm cohesion
    - High γ (1.0-10.0): Limited visibility, independent search
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: GAMMA (γ) SENSITIVITY - Light Absorption Coefficient")
    print("="*70)
    print("Testing range: γ ∈ [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]")
    print("Fixed parameters: α=0.2, β₀=1.0\n")
    
    gamma_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    dimensions = 10
    n_fireflies = 40
    max_iterations = 100
    
    # Test on both unimodal and multimodal functions
    results_sphere = []
    results_rastrigin = []
    convergence_sphere = []
    convergence_rastrigin = []
    
    for gamma in gamma_values:
        print(f"  Testing γ = {gamma}...", end='\r')
        
        # Sphere (unimodal)
        fa_sphere = FireflyAlgorithm(
            objective_func=sphere_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=1.0,
            gamma=gamma
        )
        _, best_fit_sphere, history_sphere = fa_sphere.run()
        results_sphere.append(best_fit_sphere)
        convergence_sphere.append(history_sphere)
        
        # Rastrigin (multimodal)
        fa_rastrigin = FireflyAlgorithm(
            objective_func=rastrigin_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=1.0,
            gamma=gamma
        )
        _, best_fit_rastrigin, history_rastrigin = fa_rastrigin.run()
        results_rastrigin.append(best_fit_rastrigin)
        convergence_rastrigin.append(history_rastrigin)
    
    print("\n")
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # Plot 1: Final fitness vs gamma
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(gamma_values, results_sphere, 'bo-', linewidth=2, markersize=8, label='Sphere (Unimodal)')
    ax1.semilogx(gamma_values, results_rastrigin, 'rs-', linewidth=2, markersize=8, label='Rastrigin (Multimodal)')
    ax1.set_xlabel('γ (Light Absorption Coefficient)', fontsize=11)
    ax1.set_ylabel('Final Best Fitness', fontsize=11)
    ax1.set_title('Parameter γ Sensitivity', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0.001, 0.1, alpha=0.2, color='blue', label='Good for Multimodal')
    ax1.axvspan(1.0, 10.0, alpha=0.2, color='red', label='Good for Unimodal')
    
    # Plot 2: Convergence curves for Sphere
    ax2 = fig.add_subplot(gs[0, 1])
    for i, gamma in enumerate(gamma_values):
        ax2.semilogy(convergence_sphere[i], label=f'γ={gamma}', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax2.set_title('Convergence on Sphere Function', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence curves for Rastrigin
    ax3 = fig.add_subplot(gs[0, 2])
    for i, gamma in enumerate(gamma_values):
        ax3.semilogy(convergence_rastrigin[i], label=f'γ={gamma}', alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax3.set_title('Convergence on Rastrigin Function', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fa_gamma_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✓ Gamma sensitivity analysis saved as 'fa_gamma_sensitivity.png'\n")
    plt.show()
    
    return {
        'gamma_values': gamma_values,
        'results_sphere': results_sphere,
        'results_rastrigin': results_rastrigin
    }


def analyze_alpha_sensitivity():
    """
    Analyze the effect of α (randomization parameter) on performance.
    
    α controls exploration:
    - Low α (0.01-0.1): Fine-tuning, local search
    - High α (0.4-1.0): Large jumps, global exploration
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: ALPHA (α) SENSITIVITY - Randomization Parameter")
    print("="*70)
    print("Testing range: α ∈ [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]")
    print("Fixed parameters: γ=1.0, β₀=1.0\n")
    
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    dimensions = 10
    n_fireflies = 40
    max_iterations = 100
    
    results_sphere = []
    results_rastrigin = []
    convergence_sphere = []
    convergence_rastrigin = []
    
    for alpha in alpha_values:
        print(f"  Testing α = {alpha}...", end='\r')
        
        # Sphere (unimodal)
        fa_sphere = FireflyAlgorithm(
            objective_func=sphere_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=alpha,
            beta0=1.0,
            gamma=1.0
        )
        _, best_fit_sphere, history_sphere = fa_sphere.run()
        results_sphere.append(best_fit_sphere)
        convergence_sphere.append(history_sphere)
        
        # Rastrigin (multimodal)
        fa_rastrigin = FireflyAlgorithm(
            objective_func=rastrigin_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=alpha,
            beta0=1.0,
            gamma=1.0
        )
        _, best_fit_rastrigin, history_rastrigin = fa_rastrigin.run()
        results_rastrigin.append(best_fit_rastrigin)
        convergence_rastrigin.append(history_rastrigin)
    
    print("\n")
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # Plot 1: Final fitness vs alpha
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(alpha_values, results_sphere, 'bo-', linewidth=2, markersize=8, label='Sphere (Unimodal)')
    ax1.plot(alpha_values, results_rastrigin, 'rs-', linewidth=2, markersize=8, label='Rastrigin (Multimodal)')
    ax1.set_xlabel('α (Randomization Parameter)', fontsize=11)
    ax1.set_ylabel('Final Best Fitness', fontsize=11)
    ax1.set_title('Parameter α Sensitivity', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0.01, 0.1, alpha=0.2, color='red', label='Good for Unimodal')
    ax1.axvspan(0.2, 0.5, alpha=0.2, color='blue', label='Good for Multimodal')
    
    # Plot 2: Convergence curves for Sphere
    ax2 = fig.add_subplot(gs[0, 1])
    for i, alpha in enumerate(alpha_values):
        ax2.semilogy(convergence_sphere[i], label=f'α={alpha}', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax2.set_title('Convergence on Sphere Function', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence curves for Rastrigin
    ax3 = fig.add_subplot(gs[0, 2])
    for i, alpha in enumerate(alpha_values):
        ax3.semilogy(convergence_rastrigin[i], label=f'α={alpha}', alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax3.set_title('Convergence on Rastrigin Function', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fa_alpha_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✓ Alpha sensitivity analysis saved as 'fa_alpha_sensitivity.png'\n")
    plt.show()
    
    return {
        'alpha_values': alpha_values,
        'results_sphere': results_sphere,
        'results_rastrigin': results_rastrigin
    }


def analyze_beta0_sensitivity():
    """
    Analyze the effect of β₀ (base attractiveness) on performance.
    
    β₀ controls attraction strength:
    - Low β₀ (0.1-0.5): Weak attraction, slower convergence
    - High β₀ (1.0-3.0): Strong attraction, faster convergence
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: BETA₀ (β₀) SENSITIVITY - Base Attractiveness")
    print("="*70)
    print("Testing range: β₀ ∈ [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]")
    print("Fixed parameters: γ=1.0, α=0.2\n")
    
    beta0_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    dimensions = 10
    n_fireflies = 40
    max_iterations = 100
    
    results_sphere = []
    results_rastrigin = []
    convergence_sphere = []
    convergence_rastrigin = []
    
    for beta0 in beta0_values:
        print(f"  Testing β₀ = {beta0}...", end='\r')
        
        # Sphere (unimodal)
        fa_sphere = FireflyAlgorithm(
            objective_func=sphere_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=beta0,
            gamma=1.0
        )
        _, best_fit_sphere, history_sphere = fa_sphere.run()
        results_sphere.append(best_fit_sphere)
        convergence_sphere.append(history_sphere)
        
        # Rastrigin (multimodal)
        fa_rastrigin = FireflyAlgorithm(
            objective_func=rastrigin_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=beta0,
            gamma=1.0
        )
        _, best_fit_rastrigin, history_rastrigin = fa_rastrigin.run()
        results_rastrigin.append(best_fit_rastrigin)
        convergence_rastrigin.append(history_rastrigin)
    
    print("\n")
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # Plot 1: Final fitness vs beta0
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(beta0_values, results_sphere, 'bo-', linewidth=2, markersize=8, label='Sphere (Unimodal)')
    ax1.plot(beta0_values, results_rastrigin, 'rs-', linewidth=2, markersize=8, label='Rastrigin (Multimodal)')
    ax1.set_xlabel('β₀ (Base Attractiveness)', fontsize=11)
    ax1.set_ylabel('Final Best Fitness', fontsize=11)
    ax1.set_title('Parameter β₀ Sensitivity', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0.5, 1.5, alpha=0.2, color='green', label='Recommended Range')
    
    # Plot 2: Convergence curves for Sphere
    ax2 = fig.add_subplot(gs[0, 1])
    for i, beta0 in enumerate(beta0_values):
        ax2.semilogy(convergence_sphere[i], label=f'β₀={beta0}', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax2.set_title('Convergence on Sphere Function', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence curves for Rastrigin
    ax3 = fig.add_subplot(gs[0, 2])
    for i, beta0 in enumerate(beta0_values):
        ax3.semilogy(convergence_rastrigin[i], label=f'β₀={beta0}', alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
    ax3.set_title('Convergence on Rastrigin Function', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fa_beta0_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✓ Beta₀ sensitivity analysis saved as 'fa_beta0_sensitivity.png'\n")
    plt.show()
    
    return {
        'beta0_values': beta0_values,
        'results_sphere': results_sphere,
        'results_rastrigin': results_rastrigin
    }


def analyze_parameter_interactions():
    """
    Analyze how parameter pairs interact using 2D heatmaps.
    This reveals non-obvious parameter dependencies.
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: PARAMETER INTERACTION EFFECTS")
    print("="*70)
    print("Creating 2D heatmaps for parameter pair interactions\n")
    
    dimensions = 10
    n_fireflies = 30
    max_iterations = 80
    
    # Define parameter grids
    gamma_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    beta0_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    # Interaction 1: Gamma vs Alpha on Rastrigin
    print("  Computing γ-α interaction on Rastrigin...")
    results_gamma_alpha = np.zeros((len(gamma_values), len(alpha_values)))
    
    for i, gamma in enumerate(gamma_values):
        for j, alpha in enumerate(alpha_values):
            fa = FireflyAlgorithm(
                objective_func=rastrigin_function,
                dimensions=dimensions,
                lower_bound=-5.12,
                upper_bound=5.12,
                n_fireflies=n_fireflies,
                max_iterations=max_iterations,
                alpha=alpha,
                beta0=1.0,
                gamma=gamma
            )
            _, best_fit, _ = fa.run()
            results_gamma_alpha[i, j] = best_fit
            print(f"    γ={gamma:.2f}, α={alpha:.2f}: {best_fit:.4f}", end='\r')
    
    print("\n  Computing α-β₀ interaction on Sphere...")
    results_alpha_beta0 = np.zeros((len(alpha_values), len(beta0_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, beta0 in enumerate(beta0_values):
            fa = FireflyAlgorithm(
                objective_func=sphere_function,
                dimensions=dimensions,
                lower_bound=-5.12,
                upper_bound=5.12,
                n_fireflies=n_fireflies,
                max_iterations=max_iterations,
                alpha=alpha,
                beta0=beta0,
                gamma=1.0
            )
            _, best_fit, _ = fa.run()
            results_alpha_beta0[i, j] = best_fit
            print(f"    α={alpha:.2f}, β₀={beta0:.2f}: {best_fit:.6f}", end='\r')
    
    print("\n")
    
    # Visualization
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 2, figure=fig)
    
    # Heatmap 1: Gamma vs Alpha
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(results_gamma_alpha, aspect='auto', cmap='RdYlGn_r', 
                     origin='lower', interpolation='bilinear')
    ax1.set_xticks(range(len(alpha_values)))
    ax1.set_yticks(range(len(gamma_values)))
    ax1.set_xticklabels([f'{x:.2f}' for x in alpha_values])
    ax1.set_yticklabels([f'{x:.2f}' for x in gamma_values])
    ax1.set_xlabel('α (Randomization)', fontsize=11)
    ax1.set_ylabel('γ (Light Absorption)', fontsize=11)
    ax1.set_title('γ-α Interaction on Rastrigin Function\n(Lower is Better)', 
                  fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(gamma_values)):
        for j in range(len(alpha_values)):
            text = ax1.text(j, i, f'{results_gamma_alpha[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im1, ax=ax1, label='Final Best Fitness')
    
    # Add optimal region annotation
    ax1.plot([1, 4], [0, 0], 'b*', markersize=15)
    ax1.annotate('Optimal Region\n(Low γ, Moderate α)', 
                xy=(2.5, 0.5), xytext=(4, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=9, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Heatmap 2: Alpha vs Beta0
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(results_alpha_beta0, aspect='auto', cmap='RdYlGn_r',
                     origin='lower', interpolation='bilinear')
    ax2.set_xticks(range(len(beta0_values)))
    ax2.set_yticks(range(len(alpha_values)))
    ax2.set_xticklabels([f'{x:.2f}' for x in beta0_values])
    ax2.set_yticklabels([f'{x:.2f}' for x in alpha_values])
    ax2.set_xlabel('β₀ (Base Attractiveness)', fontsize=11)
    ax2.set_ylabel('α (Randomization)', fontsize=11)
    ax2.set_title('α-β₀ Interaction on Sphere Function\n(Lower is Better)', 
                  fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(alpha_values)):
        for j in range(len(beta0_values)):
            text = ax2.text(j, i, f'{results_alpha_beta0[i, j]:.2e}',
                           ha="center", va="center", color="black", fontsize=6)
    
    plt.colorbar(im2, ax=ax2, label='Final Best Fitness')
    
    # Add optimal region annotation
    ax2.plot([3, 4], [1, 2], 'b*', markersize=15)
    ax2.annotate('Optimal Region\n(Low α, Moderate β₀)', 
                xy=(3.5, 1.5), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=9, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fa_parameter_interactions.png', dpi=150, bbox_inches='tight')
    print("✓ Parameter interaction heatmaps saved as 'fa_parameter_interactions.png'\n")
    plt.show()


def compare_across_benchmarks():
    """
    Test optimal parameter configurations across multiple benchmark functions.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: PERFORMANCE ACROSS MULTIPLE BENCHMARKS")
    print("="*70)
    print("Testing three parameter configurations on four benchmark functions\n")
    
    dimensions = 10
    n_fireflies = 40
    max_iterations = 150
    
    # Define configurations
    configs = {
        'Exploration-Focused': {'alpha': 0.4, 'beta0': 1.0, 'gamma': 0.01},
        'Balanced': {'alpha': 0.2, 'beta0': 1.0, 'gamma': 0.5},
        'Exploitation-Focused': {'alpha': 0.05, 'beta0': 1.0, 'gamma': 5.0}
    }
    
    # Define benchmark functions
    benchmarks = {
        'Sphere': (sphere_function, -5.12, 5.12, 'Unimodal, Convex'),
        'Rastrigin': (rastrigin_function, -5.12, 5.12, 'Highly Multimodal'),
        'Rosenbrock': (rosenbrock_function, -2.048, 2.048, 'Valley-Shaped'),
        'Ackley': (ackley_function, -5.0, 5.0, 'Multimodal, Flat')
    }
    
    # Run experiments
    results = {config: {bench: [] for bench in benchmarks} for config in configs}
    convergences = {config: {bench: [] for bench in benchmarks} for config in configs}
    
    for config_name, params in configs.items():
        print(f"\n  Testing {config_name} configuration...")
        print(f"    γ={params['gamma']}, α={params['alpha']}, β₀={params['beta0']}")
        
        for bench_name, (func, lb, ub, desc) in benchmarks.items():
            print(f"    - {bench_name} function...", end='\r')
            
            fa = FireflyAlgorithm(
                objective_func=func,
                dimensions=dimensions,
                lower_bound=lb,
                upper_bound=ub,
                n_fireflies=n_fireflies,
                max_iterations=max_iterations,
                **params
            )
            
            _, best_fit, history = fa.run()
            results[config_name][bench_name] = best_fit
            convergences[config_name][bench_name] = history
    
    print("\n")
    
    # Print results table
    print("="*80)
    print(f"{'Benchmark':<15} {'Type':<20} {'Exploration':<15} {'Balanced':<15} {'Exploitation':<15}")
    print("-"*80)
    
    for bench_name, (func, lb, ub, desc) in benchmarks.items():
        row = f"{bench_name:<15} {desc:<20}"
        for config_name in ['Exploration-Focused', 'Balanced', 'Exploitation-Focused']:
            val = results[config_name][bench_name]
            row += f" {val:<15.4f}"
        print(row)
    
    print("="*80)
    
    # Visualization: Convergence comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {'Exploration-Focused': 'blue', 'Balanced': 'green', 'Exploitation-Focused': 'red'}
    
    for idx, (bench_name, (func, lb, ub, desc)) in enumerate(benchmarks.items()):
        ax = axes[idx]
        
        for config_name in configs:
            ax.semilogy(convergences[config_name][bench_name], 
                       color=colors[config_name], linewidth=2, 
                       label=config_name, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Best Fitness (Log Scale)', fontsize=11)
        ax.set_title(f'{bench_name} Function - {desc}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add final fitness annotation
        textstr = 'Final Fitness:\n'
        for config_name in configs:
            val = results[config_name][bench_name]
            textstr += f'{config_name.split("-")[0]}: {val:.4f}\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('fa_benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Benchmark comparison saved as 'fa_benchmark_comparison.png'\n")
    plt.show()


def create_parameter_tuning_guide():
    """
    Create a comprehensive visual guide for parameter tuning.
    """
    print("\n" + "="*70)
    print("CREATING PARAMETER TUNING GUIDE")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Firefly Algorithm: Parameter Tuning Guide', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: Gamma effect diagram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    gamma_text = """
    γ (Gamma) - Light Absorption Coefficient
    ═══════════════════════════════════════
    Controls firefly visibility & swarm cohesion
    
    HIGH γ (1.0 - 10.0):
    • Limited visibility → local search
    • Swarm fragments → independent agents
    • Best for: UNIMODAL problems
    • Risk: Loss of global information
    
    LOW γ (0.001 - 0.1):
    • Global visibility → swarm cohesion
    • All fireflies attracted to best
    • Best for: MULTIMODAL problems  
    • Risk: Premature convergence
    
    💡 Rule: ↓ γ for complex landscapes
    """
    
    ax1.text(0.05, 0.95, gamma_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Panel 2: Alpha effect diagram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    alpha_text = """
    α (Alpha) - Randomization Parameter
    ════════════════════════════════════
    Controls random step size & exploration
    
    HIGH α (0.4 - 1.0):
    • Large random jumps
    • Escape local optima easily
    • Best for: MULTIMODAL problems
    • Risk: Unstable, no convergence
    
    LOW α (0.01 - 0.1):
    • Small fine-tuning steps
    • Smooth, stable convergence
    • Best for: UNIMODAL problems
    • Risk: Trapped in local optima
    
    💡 Rule: ↓ α over time (annealing)
    """
    
    ax2.text(0.05, 0.95, alpha_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Panel 3: Beta0 effect diagram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    beta0_text = """
    β₀ (Beta0) - Base Attractiveness
    ═════════════════════════════════
    Controls attraction strength at r=0
    
    HIGH β₀ (1.5 - 3.0):
    • Strong attraction → fast convergence
    • May overshoot
    • Risk: Oscillation, instability
    
    LOW β₀ (0.1 - 0.5):
    • Weak attraction → slow convergence
    • Stable, cautious movement
    • Risk: Slow progress
    
    STANDARD β₀ = 1.0:
    • Good default for most problems
    • Less sensitive than γ and α
    
    💡 Rule: Start with β₀ = 1.0
    """
    
    ax3.text(0.05, 0.95, beta0_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel 4: Decision tree
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    decision_text = """
    DECISION TREE: How to Choose Parameters
    ═══════════════════════════════════════
    
    Question 1: Problem Type?
    ┌─────────────────────────────────────┐
    │ UNIMODAL (Single Optimum)           │
    │ → High γ (5.0), Low α (0.05)        │
    │ → Focus on EXPLOITATION             │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │ MULTIMODAL (Many Local Optima)      │
    │ → Low γ (0.01), High α (0.4)        │
    │ → Focus on EXPLORATION              │
    └─────────────────────────────────────┘
    
    Question 2: Not Sure?
    → Start BALANCED: γ=1.0, α=0.2, β₀=1.0
    → Observe convergence
    → Adjust based on behavior
    
    💡 Universal tip: Always set β₀ = 1.0
    """
    
    ax4.text(0.05, 0.95, decision_text, transform=ax4.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # Panel 5: Recommended configurations table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create table
    table_data = [
        ['Problem Type', 'γ (gamma)', 'α (alpha)', 'β₀ (beta0)', 'Strategy', 'Examples'],
        ['Simple Unimodal', '5.0', '0.05', '1.0', 'High Exploitation', 'Sphere, Ellipsoid'],
        ['Valley-Shaped', '2.0', '0.1', '1.0', 'Moderate Exploit.', 'Rosenbrock'],
        ['Few Local Optima', '0.5', '0.2', '1.0', 'Balanced', 'Griewank'],
        ['Many Local Optima', '0.1', '0.3', '1.0', 'High Exploration', 'Rastrigin, Ackley'],
        ['Highly Complex', '0.01', '0.4', '1.0', 'Maximum Explor.', 'Schwefel, Michalewicz'],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.12, 0.12, 0.12, 0.18, 0.28])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('darkblue')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code rows
    colors = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue', 'plum']
    for i in range(1, 6):
        for j in range(6):
            table[(i, j)].set_facecolor(colors[i-1])
            table[(i, j)].set_alpha(0.7)
    
    ax5.text(0.5, 0.85, 'Recommended Parameter Configurations', 
            transform=ax5.transAxes, fontsize=13, fontweight='bold',
            ha='center', va='center')
    
    plt.savefig('fa_parameter_tuning_guide.png', dpi=150, bbox_inches='tight')
    print("✓ Parameter tuning guide saved as 'fa_parameter_tuning_guide.png'\n")
    plt.show()


def main():
    """Run comprehensive parameter sensitivity analysis"""
    print("\n" + "="*70)
    print(" "*10 + "FIREFLY ALGORITHM: PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nThis comprehensive analysis will:")
    print("  1. Analyze γ (gamma) sensitivity - Light absorption coefficient")
    print("  2. Analyze α (alpha) sensitivity - Randomization parameter")
    print("  3. Analyze β₀ (beta0) sensitivity - Base attractiveness")
    print("  4. Show parameter interaction effects (2D heatmaps)")
    print("  5. Compare performance across benchmark functions")
    print("  6. Create a parameter tuning guide")
    print("\n" + "="*70 + "\n")
    
    input("Press Enter to start the analysis...")
    
    # Run all analyses
    analyze_gamma_sensitivity()
    analyze_alpha_sensitivity()
    analyze_beta0_sensitivity()
    analyze_parameter_interactions()
    compare_across_benchmarks()
    create_parameter_tuning_guide()
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\n📊 Generated Files:")
    print("  1. fa_gamma_sensitivity.png - γ parameter analysis")
    print("  2. fa_alpha_sensitivity.png - α parameter analysis")
    print("  3. fa_beta0_sensitivity.png - β₀ parameter analysis")
    print("  4. fa_parameter_interactions.png - Parameter interaction heatmaps")
    print("  5. fa_benchmark_comparison.png - Performance across benchmarks")
    print("  6. fa_parameter_tuning_guide.png - Practical tuning guide")
    
    print("\n📝 KEY FINDINGS:")
    print("  • γ (gamma) is the MOST CRITICAL parameter")
    print("    - Controls exploration vs exploitation balance")
    print("    - Low γ for multimodal, high γ for unimodal")
    print("\n  • α (alpha) controls escape capability")
    print("    - High α helps escape local optima")
    print("    - Should decrease over time (annealing)")
    print("\n  • β₀ (beta0) is least sensitive")
    print("    - β₀ = 1.0 works well for most problems")
    print("    - Controls convergence speed")
    print("\n  • Parameter interactions are significant")
    print("    - γ and α must be tuned together")
    print("    - Wrong combination causes poor performance")
    
    print("\n🎯 PRACTICAL RECOMMENDATIONS:")
    print("  • For unknown problems: Start with γ=1.0, α=0.2, β₀=1.0")
    print("  • If stuck in local optima: Decrease γ, increase α")
    print("  • If not converging: Increase γ, decrease α")
    print("  • Always use adaptive α (decrease over iterations)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
