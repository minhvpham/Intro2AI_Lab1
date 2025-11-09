import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pso import PSO, rastrigin_plot
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.Continuous_functions import rastrigin

# --- 3. Setup and Run PSO ---

# Parameters for 2D Rastrigin
dims = 2
bounds_low = -5.12
bounds_high = 5.12
n_particles = 20
max_iterations = 100

print("Optimizing 2D Rastrigin Function...")
pso_rastrigin = PSO(obj_func=rastrigin, 
                      n_particles=n_particles, 
                      n_dims=dims, 
                      bounds_low=bounds_low,
                      bounds_high=bounds_high, 
                      w=0.8, c1=0.5, c2=0.5)
                      
# Run optimizer and get histories
pos, fit, fit_hist, pos_hist, gbest_pos_hist = pso_rastrigin.optimize(max_iterations)

print(f"Rastrigin - Best Fitness: {fit}")
print(f"Rastrigin - Best Position: {pos}\n")

# --- 4. Setup the Animation Plot ---

# Set up the plotting surface
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_tight_layout(True)

# Create the contour plot
x_plot = np.linspace(bounds_low, bounds_high, 100)
y_plot = np.linspace(bounds_low, bounds_high, 100)
X, Y = np.meshgrid(x_plot, y_plot)
Z = rastrigin_plot(X, Y)

img = ax.imshow(Z, extent=[bounds_low, bounds_high, bounds_low, bounds_high], 
                origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
contours = ax.contour(X, Y, Z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Plot the global minimum (0,0)
ax.plot(0, 0, marker='x', markersize=10, color="white", label="Global Minimum")

# Plot the initial state of the particles
initial_positions = pos_hist[0]
particles_plot = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                            marker='o', color='blue', alpha=0.7, label="Particles")

# Plot the initial gBest
initial_gbest = gbest_pos_hist[0]
gbest_plot = ax.scatter([initial_gbest[0]], [initial_gbest[1]], 
                        marker='*', s=200, color='red', alpha=1.0, label="Global Best")

# Initialize force arrows (quiver plots)
initial_vel = pso_rastrigin.velocity_history[0]
initial_inertia = pso_rastrigin.inertia_history[0]
initial_cognitive = pso_rastrigin.cognitive_history[0]
initial_social = pso_rastrigin.social_history[0]

# Velocity arrow (blue - resultant)
velocity_arrow = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                           initial_vel[:, 0], initial_vel[:, 1], 
                           color='blue', alpha=0.6, width=0.003, scale=10, 
                           label='Velocity')

# Inertia arrow (red)
inertia_arrow = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                          initial_inertia[:, 0], initial_inertia[:, 1], 
                          color='red', alpha=0.5, width=0.002, scale=10, 
                          label='Inertia')

# Cognitive force arrow (green)
cognitive_arrow = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                            initial_cognitive[:, 0], initial_cognitive[:, 1], 
                            color='green', alpha=0.5, width=0.002, scale=10, 
                            label='Cognitive')

# Social force arrow (yellow)
social_arrow = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                         initial_social[:, 0], initial_social[:, 1], 
                         color='yellow', alpha=0.5, width=0.002, scale=10, 
                         label='Social')

ax.set_xlim([bounds_low, bounds_high])
ax.set_ylim([bounds_low, bounds_high])
ax.legend(loc='upper right', fontsize=8)

# --- 5. Create Animation Function ---

def animate(i):
    """Steps of PSO: update plot artists for frame i"""
    title = f'Iteration {i:03d}'
    ax.set_title(title)
    
    # Update particle positions
    particles_plot.set_offsets(pos_hist[i])
    
    # Update gBest position
    gbest_plot.set_offsets(gbest_pos_hist[i])
    
    # Update force arrows
    current_positions = pos_hist[i]
    current_vel = pso_rastrigin.velocity_history[i]
    current_inertia = pso_rastrigin.inertia_history[i]
    current_cognitive = pso_rastrigin.cognitive_history[i]
    current_social = pso_rastrigin.social_history[i]
    
    velocity_arrow.set_offsets(current_positions)
    velocity_arrow.set_UVC(current_vel[:, 0], current_vel[:, 1])
    
    inertia_arrow.set_offsets(current_positions)
    inertia_arrow.set_UVC(current_inertia[:, 0], current_inertia[:, 1])
    
    cognitive_arrow.set_offsets(current_positions)
    cognitive_arrow.set_UVC(current_cognitive[:, 0], current_cognitive[:, 1])
    
    social_arrow.set_offsets(current_positions)
    social_arrow.set_UVC(current_social[:, 0], current_social[:, 1])
    
    return particles_plot, gbest_plot, velocity_arrow, inertia_arrow, cognitive_arrow, social_arrow

# --- 6. Run and Save Animation ---

print("Creating animation... (this may take a moment)")
anim = FuncAnimation(fig, animate, frames=len(pos_hist), interval=1000, blit=True)

# Save the animation as a GIF
# You may need to install 'imagemagick' or 'ffmpeg' for this to work 
try:
    anim.save("pso_convergence.gif", writer='imagemagick', dpi=120)
    print("Animation saved as pso_convergence.gif")
except Exception as e:
    print(f"\nCould not save animation. Error: {e}")
    print("Please ensure you have 'imagemagick' or 'ffmpeg' installed and configured for matplotlib.")

# To display the animation in environments like Jupyter:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

# Or to just show the plot window (may not be animated):
# plt.show()