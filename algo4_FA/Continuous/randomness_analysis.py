"""
Understanding Randomness in Firefly Algorithm

This script demonstrates WHY the same FA parameters produce different results
across multiple runs, and how to control/measure this variability.

Key Learning Points:
1. Sources of randomness in FA
2. Impact of random seed on reproducibility
3. Statistical analysis of stochastic algorithms
4. When variability is acceptable vs problematic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algo4_FA.Continuous.FA import FireflyAlgorithm


def sphere_function(X):
    """Simple unimodal function for testing"""
    return np.sum(X**2)


def demonstrate_randomness_sources():
    """
    Show each source of randomness independently
    """
    print("="*80)
    print(" "*20 + "RANDOMNESS SOURCES IN FIREFLY ALGORITHM")
    print("="*80)
    
    dimensions = 5
    n_fireflies = 10
    
    print("\n1. RANDOM INITIALIZATION")
    print("-"*80)
    print("Fireflies start at random positions in the search space")
    print("np.random.uniform(lower_bound, upper_bound, (n_fireflies, dimensions))\n")
    
    # Show 3 different initializations
    for trial in range(3):
        np.random.seed(trial)
        positions = np.random.uniform(-5.12, 5.12, (n_fireflies, dimensions))
        initial_fitness = np.array([sphere_function(pos) for pos in positions])
        best_initial = np.min(initial_fitness)
        
        print(f"Trial {trial+1} (seed={trial}):")
        print(f"  Best initial position: {positions[np.argmin(initial_fitness)]}")
        print(f"  Best initial fitness:  {best_initial:.6f}")
        print(f"  Distance from origin:  {np.linalg.norm(positions[np.argmin(initial_fitness)]):.6f}\n")
    
    print("💡 OBSERVATION: Different seeds → Different starting points → Different results\n")
    
    print("\n2. RANDOM MOVEMENT (Alpha Term)")
    print("-"*80)
    print("Movement equation: x_i = x_i + β(x_j - x_i) + α*(rand-0.5)*(upper-lower)")
    print("                                                  ↑ RANDOM COMPONENT\n")
    
    # Demonstrate random step variability
    alpha = 0.05
    upper_bound = 5.12
    lower_bound = -5.12
    
    print(f"With α={alpha}, showing 5 random steps in 2D:")
    for step in range(5):
        random_step = alpha * (np.random.rand(2) - 0.5) * (upper_bound - lower_bound)
        print(f"  Step {step+1}: [{random_step[0]:>7.4f}, {random_step[1]:>7.4f}]  " +
              f"(magnitude: {np.linalg.norm(random_step):.4f})")
    
    print(f"\n💡 OBSERVATION: Even LOW α={alpha} creates random steps of magnitude ~0.25")
    print("   This randomness prevents exact reproducibility without seed control\n")
    
    print("\n3. ALPHA DECAY (Adaptive Parameter)")
    print("-"*80)
    print("FA typically uses: α = α * 0.97 each iteration (adaptive annealing)")
    
    alpha_initial = 0.5
    alpha_values = [alpha_initial]
    for iteration in range(100):
        alpha_values.append(alpha_values[-1] * 0.97)
    
    print(f"Starting α = {alpha_initial}")
    print(f"After  25 iterations: α = {alpha_values[25]:.6f} ({(1-alpha_values[25]/alpha_initial)*100:.1f}% reduction)")
    print(f"After  50 iterations: α = {alpha_values[50]:.6f} ({(1-alpha_values[50]/alpha_initial)*100:.1f}% reduction)")
    print(f"After 100 iterations: α = {alpha_values[100]:.6f} ({(1-alpha_values[100]/alpha_initial)*100:.1f}% reduction)")
    
    print("\n💡 OBSERVATION: α decreases significantly → less exploration over time")
    print("   This is INTENTIONAL (exploration → exploitation shift)")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Random initializations
    ax1 = axes[0]
    for trial in range(10):
        np.random.seed(trial)
        positions = np.random.uniform(-5.12, 5.12, (20, 2))
        ax1.scatter(positions[:, 0], positions[:, 1], alpha=0.5, s=50, label=f'Seed {trial}')
    
    ax1.plot(0, 0, 'r*', markersize=20, label='Optimum')
    ax1.set_xlim([-5.12, 5.12])
    ax1.set_ylim([-5.12, 5.12])
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_title('Random Initialization\n(10 different seeds)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)
    
    # Plot 2: Random walk demonstration
    ax2 = axes[1]
    np.random.seed(42)
    position = np.array([3.0, 3.0])
    trajectory = [position.copy()]
    
    alpha = 0.1
    for _ in range(50):
        random_step = alpha * (np.random.rand(2) - 0.5) * (5.12 - (-5.12))
        position = position + random_step
        trajectory.append(position.copy())
    
    trajectory = np.array(trajectory)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', 
               edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s',
               edgecolors='black', linewidth=2, label='End', zorder=5)
    
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('x₂', fontsize=11)
    ax2.set_title(f'Random Walk (α={alpha}, 50 steps)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Alpha decay curve
    ax3 = axes[2]
    ax3.plot(alpha_values, 'b-', linewidth=2)
    ax3.axhline(alpha_initial * 0.5, color='r', linestyle='--', alpha=0.5, label='50% of initial')
    ax3.axhline(alpha_initial * 0.1, color='orange', linestyle='--', alpha=0.5, label='10% of initial')
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('α Value', fontsize=11)
    ax3.set_title(f'Alpha Decay (α *= 0.97)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fa_randomness_sources.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'fa_randomness_sources.png'\n")
    plt.show()


def compare_with_and_without_seed():
    """
    Demonstrate the dramatic difference between seeded and unseeded runs
    """
    print("\n" + "="*80)
    print(" "*25 + "SEED CONTROL DEMONSTRATION")
    print("="*80)
    
    dimensions = 10
    n_fireflies = 30
    max_iterations = 80
    n_trials = 15
    
    # Experiment 1: WITH seed control (reproducible)
    print("\n📌 Experiment 1: WITH SEED CONTROL (Reproducible)")
    print("-"*80)
    
    results_with_seed = []
    for trial in range(n_trials):
        np.random.seed(42)  # Same seed every time!
        
        fa = FireflyAlgorithm(
            objective_func=sphere_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=1.0,
            gamma=1.0
        )
        _, best_fit, _ = fa.run()
        results_with_seed.append(best_fit)
        
        if trial < 5:
            print(f"Trial {trial+1:>2}: {best_fit:.8f}")
    
    print(f"\nStatistics across {n_trials} trials:")
    print(f"  Mean:     {np.mean(results_with_seed):.8f}")
    print(f"  Std Dev:  {np.std(results_with_seed):.8f}")
    print(f"  Range:    {np.max(results_with_seed) - np.min(results_with_seed):.8f}")
    print("  ✓ ALL IDENTICAL - Perfect reproducibility!")
    
    # Experiment 2: WITHOUT seed control (stochastic)
    print("\n🎲 Experiment 2: WITHOUT SEED CONTROL (Stochastic)")
    print("-"*80)
    
    results_without_seed = []
    for trial in range(n_trials):
        # No seed set - uses current random state
        
        fa = FireflyAlgorithm(
            objective_func=sphere_function,
            dimensions=dimensions,
            lower_bound=-5.12,
            upper_bound=5.12,
            n_fireflies=n_fireflies,
            max_iterations=max_iterations,
            alpha=0.2,
            beta0=1.0,
            gamma=1.0
        )
        _, best_fit, _ = fa.run()
        results_without_seed.append(best_fit)
        
        if trial < 5:
            print(f"Trial {trial+1:>2}: {best_fit:.8f}")
    
    print(f"\nStatistics across {n_trials} trials:")
    print(f"  Mean:     {np.mean(results_without_seed):.8f}")
    print(f"  Std Dev:  {np.std(results_without_seed):.8f}  ⚠️  SIGNIFICANT VARIANCE!")
    print(f"  Range:    {np.max(results_without_seed) - np.min(results_without_seed):.8f}")
    print(f"  Min:      {np.min(results_without_seed):.8f}  (Best run)")
    print(f"  Max:      {np.max(results_without_seed):.8f}  (Worst run)")
    
    # Calculate how much worse the worst run is
    ratio = np.max(results_without_seed) / np.min(results_without_seed)
    print(f"  Worst is {ratio:.2f}x worse than best!")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Line plot showing each trial
    ax1.plot(range(1, n_trials+1), results_with_seed, 'b-o', linewidth=2, 
            markersize=8, label='With Seed (Identical)', alpha=0.8)
    ax1.plot(range(1, n_trials+1), results_without_seed, 'r-s', linewidth=2,
            markersize=6, label='Without Seed (Variable)', alpha=0.8)
    
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Final Best Fitness', fontsize=11)
    ax1.set_title('Reproducibility: With vs Without Seed Control', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    box_data = [results_with_seed, results_without_seed]
    bp = ax2.boxplot(box_data, labels=['With Seed\\n(Reproducible)', 'Without Seed\\n(Stochastic)'],
                    patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # Add scatter of individual points
    for i, data in enumerate(box_data, 1):
        y = data
        x = np.random.normal(i, 0.04, len(data))
        ax2.scatter(x, y, alpha=0.4, s=30, c='black')
    
    ax2.set_ylabel('Final Best Fitness', fontsize=11)
    ax2.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Without Seed:\\nStd = {np.std(results_without_seed):.2e}\\n' + \
                 f'Range = {np.max(results_without_seed) - np.min(results_without_seed):.2e}\\n' + \
                 f'CV = {(np.std(results_without_seed)/np.mean(results_without_seed)*100):.1f}%'
    
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fa_seed_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved as 'fa_seed_comparison.png'\n")
    plt.show()


def statistical_analysis_multiple_runs():
    """
    Proper statistical analysis: How to report results from stochastic algorithms
    """
    print("\n" + "="*80)
    print(" "*20 + "STATISTICAL ANALYSIS: BEST PRACTICES")
    print("="*80)
    
    print("\n📊 When reporting results from stochastic algorithms like FA:")
    print("   1. Run MULTIPLE independent trials (typically 10-30 runs)")
    print("   2. Use DIFFERENT random seeds for each trial")
    print("   3. Report MEAN ± STANDARD DEVIATION")
    print("   4. Report best, worst, and median performance")
    print("   5. Use statistical tests for comparison (t-test, Wilcoxon, etc.)")
    
    dimensions = 10
    n_fireflies = 40
    max_iterations = 100
    n_trials = 30
    
    # Test 3 parameter configurations
    configs = {
        'High Exploitation\\n(γ=5.0, α=0.05)': {'gamma': 5.0, 'alpha': 0.05},
        'Balanced\\n(γ=1.0, α=0.2)': {'gamma': 1.0, 'alpha': 0.2},
        'High Exploration\\n(γ=0.01, α=0.5)': {'gamma': 0.01, 'alpha': 0.5}
    }
    
    all_results = {}
    
    for config_name, params in configs.items():
        print(f"\nRunning {n_trials} trials for: {config_name.split(chr(10))[0]}...")
        results = []
        
        for trial in range(n_trials):
            np.random.seed(1000 + trial)  # Different seed each trial
            
            fa = FireflyAlgorithm(
                objective_func=sphere_function,
                dimensions=dimensions,
                lower_bound=-5.12,
                upper_bound=5.12,
                n_fireflies=n_fireflies,
                max_iterations=max_iterations,
                alpha=params['alpha'],
                beta0=1.0,
                gamma=params['gamma']
            )
            _, best_fit, _ = fa.run()
            results.append(best_fit)
        
        all_results[config_name] = results
        
        # Statistical summary
        print(f"  Mean ± Std:  {np.mean(results):.6f} ± {np.std(results):.6f}")
        print(f"  Median:      {np.median(results):.6f}")
        print(f"  Best:        {np.min(results):.6f}")
        print(f"  Worst:       {np.max(results):.6f}")
        print(f"  CV:          {(np.std(results)/np.mean(results)*100):.2f}%")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Box plot comparison
    ax1 = fig.add_subplot(gs[0, 0])
    box_data = [all_results[k] for k in configs.keys()]
    bp = ax1.boxplot(box_data, labels=[k.replace('\\n', '\n') for k in configs.keys()],
                     patch_artist=True, widths=0.5)
    
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Final Best Fitness', fontsize=11)
    ax1.set_title('Performance Distribution Comparison (30 runs)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Plot 2: Violin plot for distribution shape
    ax2 = fig.add_subplot(gs[0, 1])
    positions = [1, 2, 3]
    parts = ax2.violinplot(box_data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True, showextrema=True)
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels([k.replace('\\n', '\n') for k in configs.keys()], fontsize=9)
    ax2.set_ylabel('Final Best Fitness', fontsize=11)
    ax2.set_title('Distribution Shape (Violin Plot)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Plot 3: Mean with error bars
    ax3 = fig.add_subplot(gs[1, 0])
    means = [np.mean(all_results[k]) for k in configs.keys()]
    stds = [np.std(all_results[k]) for k in configs.keys()]
    x_pos = range(len(configs))
    
    ax3.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
           color=colors, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([k.replace('\\n', '\n') for k in configs.keys()], fontsize=9)
    ax3.set_ylabel('Mean Final Fitness', fontsize=11)
    ax3.set_title('Mean ± Std Dev (Error Bars)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax3.text(i, mean + std + max(means)*0.05, f'{mean:.2e}\\n±{std:.2e}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 4: Statistical summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = [['Metric', *[k.split('\\n')[0] for k in configs.keys()]]]
    
    metrics = ['Mean', 'Std Dev', 'Median', 'Best', 'Worst', 'Range', 'CV (%)']
    for metric in metrics:
        row = [metric]
        for config_name in configs.keys():
            data = all_results[config_name]
            if metric == 'Mean':
                val = f'{np.mean(data):.4e}'
            elif metric == 'Std Dev':
                val = f'{np.std(data):.4e}'
            elif metric == 'Median':
                val = f'{np.median(data):.4e}'
            elif metric == 'Best':
                val = f'{np.min(data):.4e}'
            elif metric == 'Worst':
                val = f'{np.max(data):.4e}'
            elif metric == 'Range':
                val = f'{np.max(data)-np.min(data):.4e}'
            elif metric == 'CV (%)':
                val = f'{(np.std(data)/np.mean(data)*100):.2f}'
            row.append(val)
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('darkblue')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i in range(1, 8):
        for j in range(4):
            if j == 0:
                table[(i, j)].set_facecolor('lightgray')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax4.text(0.5, 0.95, 'Statistical Summary (30 Independent Runs)', 
            transform=ax4.transAxes, fontsize=12, fontweight='bold',
            ha='center', va='top')
    
    plt.savefig('fa_statistical_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Statistical analysis plot saved as 'fa_statistical_analysis.png'\n")
    plt.show()
    
    # Print recommendations
    print("\n" + "="*80)
    print("📋 REPORTING BEST PRACTICES")
    print("="*80)
    print("\n✅ GOOD: 'FA achieved 0.00123 ± 0.00045 (mean ± std over 30 runs)'")
    print("✅ GOOD: Include box plots or violin plots showing distribution")
    print("✅ GOOD: Report median for skewed distributions")
    print("✅ GOOD: Use statistical tests (t-test) to compare configurations")
    print("\n❌ BAD:  'FA achieved 0.00156' (single run, not reproducible)")
    print("❌ BAD:  'FA works better' (no quantitative comparison)")
    print("❌ BAD:  Reporting only mean without standard deviation")
    print("\n" + "="*80 + "\n")


def main():
    print("\n" + "="*80)
    print(" "*15 + "UNDERSTANDING RANDOMNESS IN FIREFLY ALGORITHM")
    print("="*80)
    print("\nThis analysis explains:")
    print("  1. Why FA gives different results each run")
    print("  2. How random seed controls reproducibility")
    print("  3. Proper statistical analysis and reporting")
    print("\n" + "="*80 + "\n")
    
    input("Press Enter to start analysis...")
    
    # Run all analyses
    demonstrate_randomness_sources()
    compare_with_and_without_seed()
    statistical_analysis_multiple_runs()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n🎯 KEY TAKEAWAYS:")
    print("\n1. FA is STOCHASTIC (random) by design:")
    print("   • Random initialization of firefly positions")
    print("   • Random walk component in movement equation")
    print("   • This randomness is NECESSARY for exploration")
    
    print("\n2. To ensure REPRODUCIBILITY:")
    print("   • Set random seed: np.random.seed(42)")
    print("   • Same seed → Identical results")
    print("   • Document seed used in your experiments")
    
    print("\n3. For PROPER EVALUATION:")
    print("   • Run 10-30 independent trials with different seeds")
    print("   • Report mean ± std, not just single value")
    print("   • Use statistical tests for comparisons")
    print("   • Show distributions (box plots, violin plots)")
    
    print("\n4. WHEN IS VARIABILITY PROBLEMATIC?")
    print("   • If CV (Coefficient of Variation) > 50%")
    print("   • If best and worst runs differ by >10x")
    print("   • If you can't reliably reach acceptable solution")
    print("   → May need parameter tuning or more iterations")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
