import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import random

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.Continuous_functions import rastrigin
from FA import FireflyAlgorithm

# Configuration
D = 10
L_BOUND, U_BOUND = -5.12, 5.12

#Parameters for FA
N_FIREFLIES = 100
MAX_ITERATIONS = 200
ALPHA = 1.0
BETA0 = 1.0
GAMMA = 0.01


def plot_rastrigin_surface(lower_bound=-5.12, upper_bound=5.12, resolution=100):
    print("Generating 3D surface plot...")
    
    # Create a meshgrid 
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate Z
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))
            
    # Plot the surface [31, 32]
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Rastrigin Function 3D Surface')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.view_init(elev=30, azim=45)
    plt.savefig("rastrigin_surface_3d.png")
    plt.show()
    print("3D surface plot saved as 'rastrigin_surface_3d.png'")
# --- Visualization 2: Convergence Curves ---
def plot_convergence_curves():
    print("Running algorithms for convergence comparison...")
    
    # Set parameters for a 10-dimension problem
  
    
    # Run FA
    fa = FireflyAlgorithm(rastrigin, D, L_BOUND, U_BOUND, 
                          n_fireflies=N_FIREFLIES, max_iterations=MAX_ITERATIONS, 
                          alpha=ALPHA, beta0=BETA0, gamma=GAMMA)
    _, _, fa_history = fa.run()
   
    # Plot the curves [15, 19, 33]
    plt.figure(figsize=(12, 8))
    # Note: FE budget is 10k for all
    plt.plot(np.linspace(0, 10000, len(fa_history)), fa_history, label="Firefly Algorithm (FA)")


    plt.title('Algorithm Convergence Comparison on Rastrigin Function (d=10)')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Fitness (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("convergence_continuous.png")
    plt.show()
    print("Convergence plot saved as 'convergence_continuous.png'")

# --- Visualization 3: Animation of FA Convergence ---
def animate_firefly_algorithm():
    """Create animation showing FA convergence on 2D Rastrigin function"""
    print("Running FA for animation...")
    
    # Run FA with 2D for visualization
    D_anim = 2
    fa = FireflyAlgorithm(rastrigin, D_anim, L_BOUND, U_BOUND, 
                          n_fireflies=N_FIREFLIES, max_iterations=MAX_ITERATIONS, 
                          alpha=ALPHA, beta0=BETA0, gamma=GAMMA)
    
    # Run and get history
    best_pos, best_fit, fit_history = fa.run()
    
    # Get position history (fireflies over iterations)
    pos_hist = fa.position_history  # List of arrays: (n_fireflies, 2)
    intensity_hist = fa.intensity_history  # List of arrays: (n_fireflies,)
    best_pos_hist = fa.best_position_history  # List of arrays: (2,)
    
    # Sample every 10th iteration for animation
    step = 10
    pos_hist = pos_hist[::step]
    intensity_hist = intensity_hist[::step]
    best_pos_hist = best_pos_hist[::step]
    
    print(f"Animation will show {len(pos_hist)} frames (every {step} iterations)")
    
    # --- Set up the plotting surface ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_tight_layout(True)
    
    # Create the contour plot
    x_plot = np.linspace(L_BOUND, U_BOUND, 100)
    y_plot = np.linspace(L_BOUND, U_BOUND, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))
    
    img = ax.imshow(Z, extent=[L_BOUND, U_BOUND, L_BOUND, U_BOUND], 
                    origin='lower', cmap='viridis', alpha=0.5)
    fig.colorbar(img, ax=ax)
    contours = ax.contour(X, Y, Z, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    
    # Plot the global minimum (0,0)
    ax.plot(0, 0, marker='x', markersize=10, color="white", label="Global Minimum")
    
    # Plot the initial state of the fireflies
    initial_positions = pos_hist[0]
    initial_intensities = intensity_hist[0]
    
    # Normalize intensities for color mapping (lower fitness = brighter = warmer color)
    # We'll invert so that better (lower) fitness gets warmer colors
    norm = plt.Normalize(vmin=np.min([np.min(arr) for arr in intensity_hist]), 
                        vmax=np.max([np.min(arr) for arr in intensity_hist]) * 1.5)
    
    fireflies_plot = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                                c=-initial_intensities,  # Negative so lower fitness is warmer
                                cmap='YlOrRd', 
                                s=100, alpha=0.8, edgecolors='black', 
                                linewidths=0.5,
                                norm=norm, label="Fireflies")
    
    # Plot the initial best firefly
    initial_best = best_pos_hist[0]
    best_plot = ax.scatter([initial_best[0]], [initial_best[1]], 
                           marker='*', s=400, color='cyan', 
                           edgecolors='blue', linewidths=2,
                           alpha=1.0, label="Best Firefly", zorder=5)
    
    # Add movement arrows to show firefly movements
    movements = pos_hist[1] - pos_hist[0] if len(pos_hist) > 1 else np.zeros_like(pos_hist[0])
    movement_arrows = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                                movements[:, 0], movements[:, 1], 
                                color='orange', alpha=0.6, width=0.003, 
                                scale=20, label='Movement')
    
    ax.set_xlim([L_BOUND, U_BOUND])
    ax.set_ylim([L_BOUND, U_BOUND])
    ax.set_xlabel('x1', fontsize=10)
    ax.set_ylabel('x2', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    
    # Add text for iteration info
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # --- Animation Function ---
    def animate(i):
        """Update plot for frame i"""
        iteration_num = i * step  # Calculate actual iteration number
        title = f'Firefly Algorithm - Iteration {iteration_num}/{(len(pos_hist)-1)*step}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Update firefly positions and intensities
        current_positions = pos_hist[i]
        current_intensities = intensity_hist[i]
        
        fireflies_plot.set_offsets(current_positions)
        fireflies_plot.set_array(-current_intensities)  # Negative for color mapping
        
        # Update best firefly position
        best_plot.set_offsets([best_pos_hist[i]])
        
        # Update movement arrows
        if i < len(pos_hist) - 1:
            movements = pos_hist[i+1] - pos_hist[i]
        else:
            movements = np.zeros_like(pos_hist[i])
        
        movement_arrows.set_offsets(current_positions)
        movement_arrows.set_UVC(movements[:, 0], movements[:, 1])
        
        # Update info text
        best_fitness = np.min(current_intensities)
        avg_fitness = np.mean(current_intensities)
        info_text.set_text(f'Best Fitness: {best_fitness:.4f}\n' +
                          f'Avg Fitness: {avg_fitness:.4f}\n' +
                          f'α={ALPHA}, β₀={BETA0}, γ={GAMMA}')
        
        return fireflies_plot, best_plot, movement_arrows, info_text
    
    # --- Run and Save Animation ---
    print("Creating animation... (this may take a moment)")
    anim = animation.FuncAnimation(fig, animate, frames=len(pos_hist), 
                                   interval=1000, blit=True, repeat=True)
    
    # Save the animation as a GIF
    try:
        anim.save("fa_convergence.gif", writer='pillow', fps=10, dpi=100)
        print("Animation saved as fa_convergence.gif")
    except Exception as e:
        print(f"\nCould not save animation with pillow. Error: {e}")
        try:
            # Try with imagemagick
            anim.save("fa_convergence.gif", writer='imagemagick', dpi= 120)
            print("Animation saved as fa_convergence.gif (using imagemagick)")
        except Exception as e2:
            print(f"Could not save with imagemagick either. Error: {e2}")
            print("Please ensure you have 'pillow' or 'imagemagick' installed.")
            print("You can install pillow with: pip install pillow")
    
    plt.show()
    print("Animation display complete.")

# plot_convergence_curves()
def main():
    # Visualization 1: 3D Surface Plot
    plot_rastrigin_surface(lower_bound=-5.12, upper_bound=5.12, resolution=200)
    
    # Visualization 2: Convergence Curves
    plot_convergence_curves()
    
    # Visualization 3: Animation of FA Convergence
    animate_firefly_algorithm()
    
if __name__ == "__main__":
    main()