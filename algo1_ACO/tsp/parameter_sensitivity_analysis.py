"""
Parameter Sensitivity Analysis
===============================
This file demonstrates how different parameter combinations affect ACO performance,
showing the exploration vs. exploitation trade-off through comprehensive analysis.

Analysis Dimensions:
1. Alpha (α) - Pheromone influence
2. Beta (β) - Heuristic influence  
3. Rho (ρ) - Evaporation rate
4. Combined effects on convergence and solution quality

The goal is to visualize how each parameter affects:
- Solution quality (tour length)
- Convergence speed
- Algorithm stability
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algo1_ACO.tsp.ACO import ACO_TSP_Solver
from utils.tsp import create_cities


def run_parameter_sweep(coordinates, param_name, param_values, base_params, n_runs=3):
    """
    Run ACO with different values of a single parameter.
    
    Args:
        coordinates: City coordinates
        param_name: Name of parameter to vary ('alpha', 'beta', 'rho')
        param_values: List of values to test
        base_params: Base parameter dictionary
        n_runs: Number of runs per parameter value (for averaging)
    
    Returns:
        results: Dictionary with statistics for each parameter value
    """
    results = {
        'param_values': param_values,
        'best_lengths': [],
        'avg_lengths': [],
        'std_lengths': [],
        'convergence_iters': [],
        'convergence_histories': []
    }
    
    print(f"\n{'='*70}")
    print(f"Testing {param_name.upper()} sensitivity")
    print(f"{'='*70}")
    
    for param_val in param_values:
        print(f"\n{param_name} = {param_val:.2f}:")
        
        run_lengths = []
        run_histories = []
        run_convergence_iters = []
        
        for run in range(n_runs):
            # Update parameter
            test_params = base_params.copy()
            test_params[param_name] = param_val
            
            # Run ACO
            aco = ACO_TSP_Solver(coordinates, **test_params)
            _, length = aco.run()
            
            run_lengths.append(length)
            run_histories.append(aco.convergence_history)
            
            # Find convergence iteration (within 1% of final)
            final_val = aco.convergence_history[-1]
            conv_iter = next((i for i, v in enumerate(aco.convergence_history) 
                            if abs(v - final_val) < 0.01 * final_val), 
                           len(aco.convergence_history))
            run_convergence_iters.append(conv_iter + 1)
            
            print(f"  Run {run+1}: Length = {length:.2f}, Converged at iter {conv_iter+1}")
        
        # Store statistics
        results['best_lengths'].append(np.min(run_lengths))
        results['avg_lengths'].append(np.mean(run_lengths))
        results['std_lengths'].append(np.std(run_lengths))
        results['convergence_iters'].append(np.mean(run_convergence_iters))
        results['convergence_histories'].append(run_histories[np.argmin(run_lengths)])
        
        print(f"  → Best: {np.min(run_lengths):.2f}, Avg: {np.mean(run_lengths):.2f} ± {np.std(run_lengths):.2f}")
    
    return results


def run_full_sensitivity_analysis():
    """
    Perform comprehensive parameter sensitivity analysis.
    """
    print("="*70)
    print("ACO PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nThis analysis will test how α, β, and ρ affect ACO performance.")
    print("Testing on a 20-city TSP problem with multiple runs per configuration.\n")
    
    # Create test problem
    n_cities = 20
    coordinates = create_cities(n_cities, map_size=100, seed=456)
    
    # Base parameters (middle ground)
    base_params = {
        'n_ants': 20,
        'n_iterations': 50,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.5,
        'Q': 100
    }
    
    # Parameter ranges to test
    alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    beta_values = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rho_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    # Run parameter sweeps
    print("\n" + "🔬"*35)
    print("PARAMETER SWEEP EXPERIMENTS")
    print("🔬"*35)
    
    start_time = time.time()
    
    # 1. Alpha sweep
    base_for_alpha = base_params.copy()
    alpha_results = run_parameter_sweep(coordinates, 'alpha', alpha_values, base_for_alpha, n_runs=3)
    
    # 2. Beta sweep
    base_for_beta = base_params.copy()
    beta_results = run_parameter_sweep(coordinates, 'beta', beta_values, base_for_beta, n_runs=3)
    
    # 3. Rho sweep
    base_for_rho = base_params.copy()
    rho_results = run_parameter_sweep(coordinates, 'rho', rho_values, base_for_rho, n_runs=3)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total analysis time: {elapsed_time:.1f} seconds")
    
    # Create comprehensive visualization
    create_sensitivity_visualization(alpha_results, beta_results, rho_results, 
                                    coordinates, base_params)
    
    # Additional: 2D parameter interaction analysis
    create_2d_parameter_interaction(coordinates, base_params)
    
    return alpha_results, beta_results, rho_results


def create_sensitivity_visualization(alpha_results, beta_results, rho_results, 
                                     coordinates, base_params):
    """Create comprehensive sensitivity analysis visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Alpha Analysis
    # 1.1 Alpha - Solution Quality
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(alpha_results['param_values'], alpha_results['avg_lengths'], 
                yerr=alpha_results['std_lengths'], marker='o', linewidth=2, 
                markersize=8, capsize=5, color='blue', label='Avg ± Std')
    ax1.plot(alpha_results['param_values'], alpha_results['best_lengths'], 
            'g--', marker='s', linewidth=2, markersize=6, label='Best')
    ax1.set_xlabel('α (Pheromone Weight)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tour Length', fontsize=11, fontweight='bold')
    ax1.set_title('α Sensitivity: Solution Quality', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=base_params['alpha'], color='red', linestyle=':', linewidth=2, label='Base', alpha=0.5)
    
    # Add exploration/exploitation zones
    ax1.axvspan(0, 1.0, alpha=0.1, color='cyan', label='Exploration')
    ax1.axvspan(2.0, max(alpha_results['param_values']), alpha=0.1, color='orange', label='Exploitation')
    
    # 1.2 Alpha - Convergence Speed
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(alpha_results['param_values'], alpha_results['convergence_iters'], 
            marker='o', linewidth=2.5, markersize=8, color='purple')
    ax2.set_xlabel('α (Pheromone Weight)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Convergence Iteration', fontsize=11, fontweight='bold')
    ax2.set_title('α Sensitivity: Convergence Speed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=base_params['alpha'], color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax2.invert_yaxis()  # Lower is faster
    
    # 1.3 Alpha - Convergence Curves
    ax3 = fig.add_subplot(gs[0, 2:])
    for i, (alpha_val, history) in enumerate(zip(alpha_results['param_values'], 
                                                 alpha_results['convergence_histories'])):
        color = plt.cm.viridis(i / len(alpha_results['param_values']))
        ax3.plot(history, linewidth=2, label=f'α={alpha_val:.1f}', color=color, alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Best Tour Length', fontsize=11, fontweight='bold')
    ax3.set_title('α Sensitivity: Convergence Curves', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Beta Analysis
    # 2.1 Beta - Solution Quality
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.errorbar(beta_results['param_values'], beta_results['avg_lengths'], 
                yerr=beta_results['std_lengths'], marker='o', linewidth=2, 
                markersize=8, capsize=5, color='green', label='Avg ± Std')
    ax4.plot(beta_results['param_values'], beta_results['best_lengths'], 
            'r--', marker='s', linewidth=2, markersize=6, label='Best')
    ax4.set_xlabel('β (Heuristic Weight)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Tour Length', fontsize=11, fontweight='bold')
    ax4.set_title('β Sensitivity: Solution Quality', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=base_params['beta'], color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Add zones
    ax4.axvspan(0, 2.0, alpha=0.1, color='cyan')
    ax4.axvspan(5.0, max(beta_results['param_values']), alpha=0.1, color='orange')
    
    # 2.2 Beta - Convergence Speed
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(beta_results['param_values'], beta_results['convergence_iters'], 
            marker='o', linewidth=2.5, markersize=8, color='orange')
    ax5.set_xlabel('β (Heuristic Weight)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Convergence Iteration', fontsize=11, fontweight='bold')
    ax5.set_title('β Sensitivity: Convergence Speed', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=base_params['beta'], color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax5.invert_yaxis()
    
    # 2.3 Beta - Convergence Curves
    ax6 = fig.add_subplot(gs[1, 2:])
    for i, (beta_val, history) in enumerate(zip(beta_results['param_values'], 
                                                beta_results['convergence_histories'])):
        color = plt.cm.plasma(i / len(beta_results['param_values']))
        ax6.plot(history, linewidth=2, label=f'β={beta_val:.1f}', color=color, alpha=0.7)
    ax6.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Best Tour Length', fontsize=11, fontweight='bold')
    ax6.set_title('β Sensitivity: Convergence Curves', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, ncol=2, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Rho Analysis
    # 3.1 Rho - Solution Quality
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.errorbar(rho_results['param_values'], rho_results['avg_lengths'], 
                yerr=rho_results['std_lengths'], marker='o', linewidth=2, 
                markersize=8, capsize=5, color='red', label='Avg ± Std')
    ax7.plot(rho_results['param_values'], rho_results['best_lengths'], 
            'b--', marker='s', linewidth=2, markersize=6, label='Best')
    ax7.set_xlabel('ρ (Evaporation Rate)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Tour Length', fontsize=11, fontweight='bold')
    ax7.set_title('ρ Sensitivity: Solution Quality', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.axvline(x=base_params['rho'], color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Add zones
    ax7.axvspan(0, 0.3, alpha=0.1, color='orange', label='Exploitation')
    ax7.axvspan(0.7, 1.0, alpha=0.1, color='cyan', label='Exploration')
    
    # 3.2 Rho - Convergence Speed
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(rho_results['param_values'], rho_results['convergence_iters'], 
            marker='o', linewidth=2.5, markersize=8, color='teal')
    ax8.set_xlabel('ρ (Evaporation Rate)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Convergence Iteration', fontsize=11, fontweight='bold')
    ax8.set_title('ρ Sensitivity: Convergence Speed', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.axvline(x=base_params['rho'], color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax8.invert_yaxis()
    
    # 3.3 Rho - Convergence Curves
    ax9 = fig.add_subplot(gs[2, 2:])
    for i, (rho_val, history) in enumerate(zip(rho_results['param_values'], 
                                               rho_results['convergence_histories'])):
        color = plt.cm.coolwarm(i / len(rho_results['param_values']))
        ax9.plot(history, linewidth=2, label=f'ρ={rho_val:.1f}', color=color, alpha=0.7)
    ax9.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Best Tour Length', fontsize=11, fontweight='bold')
    ax9.set_title('ρ Sensitivity: Convergence Curves', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8, ncol=2, loc='best')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('ACO PARAMETER SENSITIVITY ANALYSIS: α, β, ρ Effects on Performance\n' + 
                 'Understanding the Exploration vs. Exploitation Trade-off',
                 fontsize=15, fontweight='bold', y=0.998)
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'parameter_sensitivity_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Main sensitivity analysis saved to: {output_path}")
    
    plt.show()


def create_2d_parameter_interaction(coordinates, base_params):
    """
    Create 2D heatmaps showing parameter interactions.
    Tests combinations of two parameters at a time.
    """
    print("\n" + "="*70)
    print("2D PARAMETER INTERACTION ANALYSIS")
    print("="*70)
    print("\nTesting parameter combinations (this may take a while)...\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define parameter grids (smaller for speed)
    alpha_range = np.linspace(0.5, 3.0, 6)
    beta_range = np.linspace(1.0, 7.0, 6)
    rho_range = np.linspace(0.2, 0.8, 6)
    
    # 1. Alpha vs Beta (fixed rho)
    print("Testing α vs β interaction...")
    results_ab = np.zeros((len(alpha_range), len(beta_range)))
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            params = base_params.copy()
            params['alpha'] = alpha
            params['beta'] = beta
            params['n_iterations'] = 30  # Shorter for speed
            
            aco = ACO_TSP_Solver(coordinates, **params)
            _, length = aco.run()
            results_ab[i, j] = length
            print(f"  α={alpha:.2f}, β={beta:.2f} → {length:.2f}")
    
    im1 = axes[0].imshow(results_ab, cmap='viridis', aspect='auto', origin='lower')
    axes[0].set_xlabel('β (Heuristic Weight)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('α (Pheromone Weight)', fontsize=11, fontweight='bold')
    axes[0].set_title('α vs β Interaction\n(Lower is better)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(beta_range)))
    axes[0].set_yticks(range(len(alpha_range)))
    axes[0].set_xticklabels([f'{b:.1f}' for b in beta_range])
    axes[0].set_yticklabels([f'{a:.1f}' for a in alpha_range])
    plt.colorbar(im1, ax=axes[0], label='Tour Length')
    
    # 2. Alpha vs Rho (fixed beta)
    print("\nTesting α vs ρ interaction...")
    results_ar = np.zeros((len(alpha_range), len(rho_range)))
    
    for i, alpha in enumerate(alpha_range):
        for j, rho in enumerate(rho_range):
            params = base_params.copy()
            params['alpha'] = alpha
            params['rho'] = rho
            params['n_iterations'] = 30
            
            aco = ACO_TSP_Solver(coordinates, **params)
            _, length = aco.run()
            results_ar[i, j] = length
            print(f"  α={alpha:.2f}, ρ={rho:.2f} → {length:.2f}")
    
    im2 = axes[1].imshow(results_ar, cmap='viridis', aspect='auto', origin='lower')
    axes[1].set_xlabel('ρ (Evaporation Rate)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('α (Pheromone Weight)', fontsize=11, fontweight='bold')
    axes[1].set_title('α vs ρ Interaction\n(Lower is better)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(rho_range)))
    axes[1].set_yticks(range(len(alpha_range)))
    axes[1].set_xticklabels([f'{r:.1f}' for r in rho_range])
    axes[1].set_yticklabels([f'{a:.1f}' for a in alpha_range])
    plt.colorbar(im2, ax=axes[1], label='Tour Length')
    
    # 3. Beta vs Rho (fixed alpha)
    print("\nTesting β vs ρ interaction...")
    results_br = np.zeros((len(beta_range), len(rho_range)))
    
    for i, beta in enumerate(beta_range):
        for j, rho in enumerate(rho_range):
            params = base_params.copy()
            params['beta'] = beta
            params['rho'] = rho
            params['n_iterations'] = 30
            
            aco = ACO_TSP_Solver(coordinates, **params)
            _, length = aco.run()
            results_br[i, j] = length
            print(f"  β={beta:.2f}, ρ={rho:.2f} → {length:.2f}")
    
    im3 = axes[2].imshow(results_br, cmap='viridis', aspect='auto', origin='lower')
    axes[2].set_xlabel('ρ (Evaporation Rate)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('β (Heuristic Weight)', fontsize=11, fontweight='bold')
    axes[2].set_title('β vs ρ Interaction\n(Lower is better)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(rho_range)))
    axes[2].set_yticks(range(len(beta_range)))
    axes[2].set_xticklabels([f'{r:.1f}' for r in rho_range])
    axes[2].set_yticklabels([f'{b:.1f}' for b in beta_range])
    plt.colorbar(im3, ax=axes[2], label='Tour Length')
    
    plt.suptitle('2D Parameter Interaction Heatmaps: Finding Optimal Combinations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '2d_parameter_interaction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 2D interaction analysis saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("ACO PARAMETER SENSITIVITY ANALYSIS")
    print("🔬" * 35 + "\n")
    
    print("This comprehensive analysis will demonstrate:")
    print("  1. How α affects pheromone trust and exploitation")
    print("  2. How β affects heuristic greediness")
    print("  3. How ρ affects exploration through evaporation")
    print("  4. How these parameters interact with each other")
    print("\nNote: This analysis runs multiple experiments and may take several minutes.")
    
    input("\nPress Enter to start the analysis...")
    
    alpha_results, beta_results, rho_results = run_full_sensitivity_analysis()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FROM SENSITIVITY ANALYSIS:")
    print("="*70)
    print("\n📊 ALPHA (α) - Pheromone Weight:")
    print("  • Low α → More exploration, less trust in swarm")
    print("  • High α → More exploitation, strong trust in learned paths")
    print(f"  • Best α in this test: {alpha_results['param_values'][np.argmin(alpha_results['best_lengths'])]:.2f}")
    
    print("\n📊 BETA (β) - Heuristic Weight:")
    print("  • Low β → Less greedy, more random exploration")
    print("  • High β → Very greedy, strong preference for nearby cities")
    print(f"  • Best β in this test: {beta_results['param_values'][np.argmin(beta_results['best_lengths'])]:.2f}")
    
    print("\n📊 RHO (ρ) - Evaporation Rate:")
    print("  • Low ρ → Slow forgetting, strong exploitation")
    print("  • High ρ → Fast forgetting, strong exploration")
    print(f"  • Best ρ in this test: {rho_results['param_values'][np.argmin(rho_results['best_lengths'])]:.2f}")
    
    print("\n🎯 OPTIMAL BALANCE:")
    print("  The best parameters depend on problem complexity:")
    print("  • Simple problems → High exploitation (high α, β; low ρ)")
    print("  • Complex problems → More exploration (low α, β; high ρ)")
    print("  • The 'sweet spot' balances both for your specific problem")
    
    print("="*70 + "\n")
