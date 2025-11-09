import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --- Visualization Functions ---
def plot_realtime_iteration_with_ants(coordinates, distances, pheromones, 
                                      ant_movements_list, ant_lengths, 
                                      iteration, best_path=None, best_length=np.inf):
    """
    Visualize a single iteration showing all ants' movements and pheromone levels.
    
    Args:
        coordinates: City coordinates
        distances: Distance matrix
        pheromones: Current pheromone matrix
        ant_movements_list: List of movement steps for each ant
        ant_lengths: Path lengths for each ant
        iteration: Current iteration number
        best_path: Best path found so far
        best_length: Best length found so far
    """
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # === Left plot: All ants' movements ===
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=150, 
               zorder=5, edgecolors='darkblue', linewidth=2, label="Cities")
    
    # Plot each ant's path
    colors = plt.cm.tab20(np.linspace(0, 1, len(ant_movements_list)))
    for ant_idx, (movements, length) in enumerate(zip(ant_movements_list, ant_lengths)):
        for step_idx, (from_city, to_city) in enumerate(movements):
            if from_city != to_city:  # Skip start position
                x = [coordinates[from_city, 0], coordinates[to_city, 0]]
                y = [coordinates[from_city, 1], coordinates[to_city, 1]]
                alpha = 0.3 + 0.7 * (step_idx / max(len(movements), 1))  # Gradient effect
                ax1.plot(x, y, color=colors[ant_idx], linewidth=1, alpha=alpha, zorder=2)
    
    # Plot arrows for ant movements (sample every 3 steps for clarity)
    for ant_idx, (movements, length) in enumerate(zip(ant_movements_list, ant_lengths)):
        path_coords = []
        for from_city, to_city in movements:
            if from_city != to_city:
                path_coords.append([coordinates[from_city], coordinates[to_city]])
        
        if path_coords:
            for i in range(0, len(path_coords), max(1, len(path_coords) // 3)):
                from_c, to_c = path_coords[i]
                dx = to_c[0] - from_c[0]
                dy = to_c[1] - from_c[1]
                ax1.arrow(from_c[0], from_c[1], dx*0.7, dy*0.7,
                         head_width=1.5, head_length=1, fc=colors[ant_idx], 
                         ec=colors[ant_idx], alpha=0.6, zorder=3)
    
    # Highlight best path in this iteration
    if best_path is not None and len(best_path) > 1:
        path_coords = coordinates[best_path, :]
        ax1.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                linewidth=3, linestyle='--', marker='o', markersize=8, 
                label=f"Best in Iteration (L: {best_length:.2f})", zorder=10)
        ax1.scatter(coordinates[best_path[0], 0], coordinates[best_path[0], 1],
                   c='gold', s=300, marker='*', zorder=11, edgecolors='orange', linewidth=2)
    
    ax1.set_title(f"Iteration {iteration} - Ant Movements (All {len(ant_movements_list)} Ants)", 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend(fontsize=9, loc='upper right')
    
    # === Right plot: Pheromone heatmap ===
    ax2 = fig.add_subplot(gs[1])
    
    # Normalize pheromones for visualization
    pheromone_normalized = pheromones.copy()
    pheromone_max = np.max(pheromone_normalized)
    if pheromone_max > 0:
        pheromone_normalized = pheromone_normalized / pheromone_max
    
    im = ax2.imshow(pheromone_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Pheromone Intensity on Edges", fontsize=12, fontweight='bold')
    ax2.set_xlabel("To City")
    ax2.set_ylabel("From City")
    cbar = plt.colorbar(im, ax=ax2, label="Pheromone Level (Normalized)")
    
    fig.suptitle(f"ACO Iteration {iteration} - Ant Movements & Pheromone Trails", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def plot_iteration_with_ants_and_pheromones(coordinates, distances, pheromones, 
                                            all_paths, all_lengths, iteration, 
                                            best_path=None, best_length=np.inf):
    """Visualize a single iteration showing all ants' paths and pheromone levels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Ant paths ===
    ax1 = axes[0]
    ax1.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, 
               zorder=5, edgecolors='darkblue', linewidth=2, label="Cities")
    
    # Plot each ant's path with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_paths)))
    for ant_idx, (path, length) in enumerate(zip(all_paths, all_lengths)):
        path_coords = coordinates[path, :]
        ax1.plot(path_coords[:, 0], path_coords[:, 1], color=colors[ant_idx], 
                linewidth=1.5, alpha=0.6, label=f"Ant {ant_idx + 1} (L: {length:.1f})")
    
    # Highlight best path in this iteration
    if best_path is not None and len(best_path) > 1:
        path_coords = coordinates[best_path, :]
        ax1.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                linewidth=3, linestyle='--', marker='o', markersize=6, 
                label=f"Best in Iteration (L: {best_length:.2f})", zorder=10)
    
    ax1.set_title(f"Iteration {iteration} - All Ants' Paths", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    if len(all_paths) <= 5:
        ax1.legend(fontsize=8, loc='upper left')
    
    # === Right plot: Pheromone heatmap ===
    ax2 = axes[1]
    
    # Normalize pheromones for visualization
    pheromone_normalized = pheromones.copy()
    pheromone_max = np.max(pheromone_normalized)
    if pheromone_max > 0:
        pheromone_normalized = pheromone_normalized / pheromone_max
    
    im = ax2.imshow(pheromone_normalized, cmap='YlOrRd', aspect='auto')
    ax2.set_title("Pheromone Levels on Edges", fontsize=12, fontweight='bold')
    ax2.set_xlabel("City")
    ax2.set_ylabel("City")
    plt.colorbar(im, ax=ax2, label="Pheromone Level")
    
    fig.suptitle(f"ACO Iteration {iteration} - Ant Movements & Pheromones", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def plot_construction_process(coordinates, paths_history, title="Ant Tour Construction Process"):
    """Visualize step-by-step construction of a single ant's tour."""
    n_steps = len(paths_history)
    cols = 5
    rows = (n_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()
    
    for step, path in enumerate(paths_history):
        ax = axes[step]
        
        # Plot all cities
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='lightblue', s=100, zorder=2, edgecolors='blue')
        
        # Plot the current partial path
        if len(path) > 1:
            path_coords = coordinates[path, :]
            ax.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                   linewidth=2, linestyle='-', marker='o', markersize=6, zorder=3)
            
            # Highlight current city with star
            ax.scatter(coordinates[path[-1], 0], coordinates[path[-1], 1], 
                      c='green', s=300, marker='*', zorder=4, label="Current", edgecolors='darkgreen', linewidth=2)
            
            # Highlight start city
            ax.scatter(coordinates[path[0], 0], coordinates[path[0], 1], 
                      c='orange', s=150, marker='s', zorder=4, label="Start", edgecolors='darkorange', linewidth=2)
        
        ax.set_title(f"Step {step + 1}: {len(path)} cities visited", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(coordinates[:, 0].min() - 5, coordinates[:, 0].max() + 5)
        ax.set_ylim(coordinates[:, 1].min() - 5, coordinates[:, 1].max() + 5)
        if step == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide extra subplots
    for idx in range(len(paths_history), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_iteration_progress(coordinates, iteration_best_paths, iteration_best_lengths, 
                           title="ACO Iteration Progress (Best Path Each Iteration)"):
    """Visualize the best path found at each iteration."""
    n_iterations = len(iteration_best_paths)
    cols = 5
    rows = (n_iterations + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()
    
    for iteration, (path, length) in enumerate(zip(iteration_best_paths, iteration_best_lengths)):
        ax = axes[iteration]
        
        # Plot all cities
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='lightblue', s=80, 
                  zorder=2, edgecolors='blue', linewidth=1)
        
        # Plot the best path for this iteration
        if path is not None and len(path) > 1:
            path_coords = coordinates[path, :]
            ax.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                   linewidth=2, linestyle='-', marker='o', markersize=5, zorder=3)
            
            # Highlight start/end
            ax.scatter(coordinates[path[0], 0], coordinates[path[0], 1], 
                      c='green', s=200, marker='s', zorder=4, edgecolors='darkgreen', linewidth=2)
        
        ax.set_title(f"Iteration {iteration + 1}\nLength: {length:.2f}", 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(coordinates[:, 0].min() - 5, coordinates[:, 0].max() + 5)
        ax.set_ylim(coordinates[:, 1].min() - 5, coordinates[:, 1].max() + 5)
    
    # Hide extra subplots
    for idx in range(len(iteration_best_paths), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
def plot_solution(coordinates, best_path, best_length, title="ACO Solution for TSP"):
    """Visualize the TSP solution using Matplotlib."""
    plt.figure(figsize=(10, 8))
    
    # Plot cities as dots
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=50, label="Cities")
    
    # Plot the best path as lines
    if best_path:
        # Reorder coordinates based on the best_path
        path_coords = coordinates[best_path, :]
        plt.plot(path_coords[:, 0], path_coords[:, 1], c='red', 
                 linewidth=1, linestyle='-', label="Best Path")
    
    plt.title(f"{title} (Length: {best_length:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(history):
    """Plot the convergence of the best solution cost over iterations."""
    plt.figure(figsize=(12, 7))
    
    # Main convergence plot
    plt.subplot(2, 2, 1)
    plt.plot(history, linewidth=2, color='#2E86AB', label="Best Cost")
    plt.title("ACOR Convergence Over Iterations", fontsize=12, fontweight='bold')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Best Cost", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Log-scale plot (better for seeing convergence when values get very small)
    plt.subplot(2, 2, 2)
    plt.semilogy(history, linewidth=2, color='#A23B72', label="Best Cost (Log Scale)")
    plt.title("ACOR Convergence (Log Scale)", fontsize=12, fontweight='bold')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Best Cost (Log Scale)", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, which='both')
    
    # Improvement rate plot
    plt.subplot(2, 2, 3)
    if len(history) > 1:
        improvements = [0]  # First iteration has no improvement
        for i in range(1, len(history)):
            improvement = history[i-1] - history[i]
            improvements.append(improvement)
        plt.plot(improvements, linewidth=2, color='#F18F01', label="Improvement per Iteration")
        plt.title("Improvement Rate", fontsize=12, fontweight='bold')
        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel("Cost Improvement", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Statistics panel
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Convergence Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━
    Initial Cost: {history[0]:.6f}
    Final Cost: {history[-1]:.6f}
    Total Improvement: {history[0] - history[-1]:.6f}
    Improvement %: {((history[0] - history[-1]) / history[0] * 100):.2f}%
    
    Iterations: {len(history)}
    Best Found at Iter: {np.argmin(history) + 1}
    
    Min Cost: {np.min(history):.6f}
    Avg Cost (Last 10): {np.mean(history[-10:]):.6f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def plot_rastrigin_surface_2d(bounds=(-5.12, 5.12), resolution=100):
    """Plot the 2D Rastrigin function surface."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Rastrigin function for 2D
    A = 10
    Z = 2 * A + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('f(X, Y)', fontsize=10)
    ax1.set_title('Rastrigin Function - 3D Surface', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    ax1.view_init(elev=30, azim=45)
    
    # 2D contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_title('Rastrigin Function - Contour Plot', fontsize=12, fontweight='bold')
    ax2.plot(0, 0, 'r*', markersize=20, label='Global Minimum (0, 0)', zorder=10)
    ax2.legend(fontsize=9)
    fig.colorbar(contour, ax=ax2, label='f(X, Y)')
    
    plt.tight_layout()
    plt.show()


def plot_convergence_with_solutions(history, archive_history=None, best_solution=None, bounds=(-5.12, 5.12)):
    """
    Enhanced convergence visualization with solution evolution for 2D problems.
    
    Args:
        history: List of best costs over iterations
        archive_history: List of archives at different iterations (optional)
        best_solution: Final best solution found
        bounds: Search space bounds
    """
    if archive_history is None or len(archive_history) == 0:
        # Fallback to basic plot if no archive history
        plot_convergence(history)
        return
    
    # Select key iterations to display (start, 25%, 50%, 75%, end)
    # Note: archive_history has n+1 elements (initial + n iterations)
    # while history has n elements (one per iteration)
    n_archives = len(archive_history)
    n_history = len(history)
    
    # Create key iteration indices based on archive history length
    key_iterations = [0, n_archives//4, n_archives//2, 3*n_archives//4, n_archives-1]
    key_iterations = [i for i in key_iterations if i < n_archives]
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, len(key_iterations), hspace=0.3, wspace=0.3)
    
    # Top row: Convergence plot spanning all columns
    ax_conv = fig.add_subplot(gs[0, :])
    ax_conv.plot(history, linewidth=2, color='#2E86AB', label="Best Cost")
    ax_conv.set_title("ACOR Convergence - Solution Evolution", fontsize=14, fontweight='bold')
    ax_conv.set_xlabel("Iteration", fontsize=11)
    ax_conv.set_ylabel("Best Cost", fontsize=11)
    ax_conv.grid(True, alpha=0.3)
    
    # Highlight key iterations on convergence plot
    # Map archive indices to history indices (archive_idx - 1 for iterations > 0)
    for archive_idx in key_iterations:
        if archive_idx > 0:  # Skip initial archive (before iteration 0)
            history_idx = archive_idx - 1
            if history_idx < n_history:
                ax_conv.axvline(x=history_idx, color='red', linestyle='--', alpha=0.3)
                ax_conv.plot(history_idx, history[history_idx], 'ro', markersize=8)
    ax_conv.legend(fontsize=10)
    
    # Create Rastrigin contour background for 2D visualization
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    A = 10
    Z = 2 * A + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
    
    # Middle row: 2D scatter plots of archive solutions at key iterations
    for col_idx, iter_idx in enumerate(key_iterations):
        ax = fig.add_subplot(gs[1, col_idx])
        
        # Plot contour background
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
        ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.2, linewidths=0.5)
        
        # Plot archive solutions
        archive = archive_history[iter_idx]
        if archive.shape[1] >= 2:  # Only plot if at least 2D
            ax.scatter(archive[:, 0], archive[:, 1], c='red', s=50, 
                      alpha=0.6, edgecolors='darkred', linewidth=1, label='Archive Solutions')
            # Highlight best solution
            # Get the cost from the archive itself
            best_cost_at_iter = archive[0, -1]
            ax.scatter(archive[0, 0], archive[0, 1], c='yellow', s=200, 
                      marker='*', edgecolors='orange', linewidth=2, 
                      label=f'Best (Cost: {best_cost_at_iter:.3f})', zorder=10)
        
        # Mark global optimum
        ax.plot(0, 0, 'w*', markersize=15, label='Global Min', zorder=5)
        
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        # Show iteration number (0 means initial, then 1, 2, 3...)
        iter_label = "Initial" if iter_idx == 0 else f"Iteration {iter_idx}"
        ax.set_title(iter_label, fontsize=10, fontweight='bold')
        ax.set_xlabel('X₁', fontsize=9)
        ax.set_ylabel('X₂', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
    
    # Bottom row: Archive diversity metrics
    for col_idx, iter_idx in enumerate(key_iterations):
        ax = fig.add_subplot(gs[2, col_idx])
        
        archive = archive_history[iter_idx]
        costs = archive[:, -1]
        
        # Plot cost distribution in archive
        ax.bar(range(len(costs)), sorted(costs), color='steelblue', alpha=0.7)
        iter_label = "Initial" if iter_idx == 0 else f"Iter {iter_idx}"
        ax.set_title(f"Archive Quality\n({iter_label})", fontsize=10, fontweight='bold')
        ax.set_xlabel('Solution Rank', fontsize=8)
        ax.set_ylabel('Cost', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)
        
        # Add statistics
        stats_text = f"Min: {np.min(costs):.2f}\nAvg: {np.mean(costs):.2f}\nMax: {np.max(costs):.2f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
               fontsize=7, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"ACOR Algorithm: Solution Space Exploration and Convergence\nFinal Best Cost: {history[-1]:.6f}", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.show()


def plot_animated_convergence(history, archive_history=None, bounds=(-5.12, 5.12), 
                              pause_time=1.0, cost_function_name="Rastrigin"):
    """
    Animated convergence visualization showing each iteration.
    
    Args:
        history: List of best costs over iterations
        archive_history: List of archives at different iterations (optional, for 2D)
        bounds: Search space bounds (for 2D visualization)
        pause_time: Time to display each iteration in seconds
        cost_function_name: Name of the cost function being optimized
    """
    plt.ion()  # Turn on interactive mode
    
    # Determine if we can show 2D plots
    show_2d = (archive_history is not None and len(archive_history) > 0 and 
               archive_history[0].shape[1] >= 3)  # At least 2D + cost
    
    if show_2d:
        # Create figure with subplots for 2D visualization
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Top left: Convergence curve
        ax_conv = fig.add_subplot(gs[0, :2])
        
        # Top right: Statistics
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        
        # Bottom left: 2D solution space
        ax_space = fig.add_subplot(gs[1, 0])
        
        # Bottom middle: Archive quality
        ax_archive = fig.add_subplot(gs[1, 1])
        
        # Bottom right: Improvement rate
        ax_improve = fig.add_subplot(gs[1, 2])
        
        # Prepare contour for solution space (for Rastrigin-like functions)
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        A = 10
        Z = 2 * A + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
        
    else:
        # Simpler layout without 2D visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_conv = axes[0, 0]
        ax_stats = axes[0, 1]
        ax_stats.axis('off')
        ax_improve = axes[1, 0]
        ax_archive = axes[1, 1]
    
    # Iterate through each iteration
    for iter_num in range(len(history)):
        # Clear all axes
        ax_conv.clear()
        ax_stats.clear()
        ax_stats.axis('off')
        ax_improve.clear()
        if show_2d:
            ax_space.clear()
        ax_archive.clear()
        
        # Get data up to current iteration
        current_history = history[:iter_num + 1]
        
        # ===== Convergence Curve =====
        ax_conv.plot(current_history, linewidth=2.5, color='#2E86AB', marker='o', 
                    markersize=4, label="Best Cost")
        ax_conv.plot(iter_num, current_history[-1], 'ro', markersize=10, zorder=10)
        ax_conv.set_title(f"ACOR Convergence - Iteration {iter_num + 1}/{len(history)}", 
                         fontsize=13, fontweight='bold')
        ax_conv.set_xlabel("Iteration", fontsize=11)
        ax_conv.set_ylabel("Best Cost", fontsize=11)
        ax_conv.grid(True, alpha=0.3)
        ax_conv.legend(fontsize=10)
        
        # Set consistent y-axis limits
        if len(current_history) > 1:
            y_min = min(current_history) * 0.9 if min(current_history) > 0 else min(current_history) * 1.1
            y_max = max(current_history) * 1.1 if max(current_history) > 0 else max(current_history) * 0.9
            ax_conv.set_ylim(y_min, y_max)
        
        # ===== Statistics Panel =====
        current_best = current_history[-1]
        initial_cost = history[0]
        improvement = initial_cost - current_best
        improvement_pct = (improvement / initial_cost * 100) if initial_cost != 0 else 0
        
        if iter_num > 0:
            recent_improvement = current_history[iter_num - 1] - current_best
        else:
            recent_improvement = 0
        
        stats_text = f"""
        Iteration: {iter_num + 1} / {len(history)}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━
        Current Best: {current_best:.6f}
        Initial Cost: {initial_cost:.6f}
        
        Total Improvement: {improvement:.6f}
        Improvement %: {improvement_pct:.2f}%
        
        Last Step Δ: {recent_improvement:.6f}
        
        Function: {cost_function_name}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center', 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # ===== Improvement Rate =====
        if iter_num > 0:
            improvements = [0]
            for i in range(1, len(current_history)):
                improvements.append(current_history[i-1] - current_history[i])
            
            colors = ['green' if imp >= 0 else 'red' for imp in improvements]
            ax_improve.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            ax_improve.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax_improve.set_title("Cost Improvement per Iteration", fontsize=11, fontweight='bold')
            ax_improve.set_xlabel("Iteration", fontsize=9)
            ax_improve.set_ylabel("Cost Reduction", fontsize=9)
            ax_improve.grid(True, alpha=0.3, axis='y')
        
        # ===== 2D Solution Space (if available) =====
        if show_2d and iter_num < len(archive_history):
            # Plot contour
            ax_space.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
            ax_space.contour(X, Y, Z, levels=10, colors='white', alpha=0.2, linewidths=0.5)
            
            # Plot archive solutions
            archive = archive_history[iter_num]
            if archive.shape[1] >= 3:
                ax_space.scatter(archive[:, 0], archive[:, 1], c='red', s=50, 
                               alpha=0.6, edgecolors='darkred', linewidth=1, 
                               label='Archive Solutions')
                # Highlight best
                ax_space.scatter(archive[0, 0], archive[0, 1], c='yellow', s=300, 
                               marker='*', edgecolors='orange', linewidth=2, 
                               label='Best Solution', zorder=10)
            
            # Mark global optimum
            ax_space.plot(0, 0, 'w*', markersize=20, label='Global Min', zorder=5,
                         markeredgecolor='black', markeredgewidth=1.5)
            
            ax_space.set_xlim(bounds[0], bounds[1])
            ax_space.set_ylim(bounds[0], bounds[1])
            ax_space.set_title("Solution Space", fontsize=11, fontweight='bold')
            ax_space.set_xlabel('X₁', fontsize=9)
            ax_space.set_ylabel('X₂', fontsize=9)
            ax_space.legend(fontsize=8, loc='upper right')
            ax_space.grid(True, alpha=0.2)
            
            # Archive quality distribution
            costs = archive[:, -1]
            ax_archive.bar(range(len(costs)), sorted(costs), color='steelblue', alpha=0.7)
            ax_archive.set_title("Archive Cost Distribution", fontsize=11, fontweight='bold')
            ax_archive.set_xlabel('Solution Rank', fontsize=9)
            ax_archive.set_ylabel('Cost', fontsize=9)
            ax_archive.grid(True, alpha=0.3, axis='y')
        else:
            # Just show a text placeholder if no 2D data
            if not show_2d:
                ax_archive.text(0.5, 0.5, f"Iteration {iter_num + 1}\n\n"
                               f"Best Cost: {current_best:.6f}", 
                               ha='center', va='center', fontsize=12,
                               transform=ax_archive.transAxes)
                ax_archive.set_title("Current Status", fontsize=11, fontweight='bold')
        
        fig.suptitle(f"ACOR Animated Convergence - {cost_function_name} Function", 
                    fontsize=14, fontweight='bold')
        
        plt.draw()
        plt.pause(pause_time)
    
    plt.ioff()  # Turn off interactive mode
    print("\nAnimation complete! Showing final result...")
    plt.show()


def plot_animated_convergence_simple(history, pause_time=1.0, cost_function_name="Unknown"):
    """
    Simple animated convergence plot for quick visualization.
    
    Args:
        history: List of best costs over iterations
        pause_time: Time to display each iteration in seconds
        cost_function_name: Name of the cost function
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for iter_num in range(len(history)):
        ax.clear()
        current_history = history[:iter_num + 1]
        
        # Plot convergence
        ax.plot(current_history, linewidth=2.5, color='#2E86AB', marker='o', 
               markersize=5, label="Best Cost")
        ax.plot(iter_num, current_history[-1], 'ro', markersize=12, zorder=10,
               label=f"Current: {current_history[-1]:.6f}")
        
        ax.set_title(f"ACOR Convergence - Iteration {iter_num + 1}/{len(history)}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Cost", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add progress bar effect
        progress = (iter_num + 1) / len(history)
        ax.text(0.02, 0.98, f"Progress: {progress*100:.1f}%", 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.draw()
        plt.pause(pause_time)
    
    plt.ioff()
    print("\nAnimation complete!")
    plt.show()
    plt.show()