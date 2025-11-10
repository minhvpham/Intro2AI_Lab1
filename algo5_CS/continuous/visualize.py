import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------- Configurable parameters ---------------------- #
N = 15              # Number of nests (population size) - (Lower for visual clarity)
D = 2               # Dimension of the problem (MUST BE 2 FOR VISUALIZATION)
LB, UB = -10, 10    # Lower and upper bounds (Smaller range for a better plot)
pa = 0.3            # Discovery rate of alien eggs (fraction of nests to replace)
alpha = 0.1         # Step size scaling factor (Slightly larger for faster visible movement)
MaxGen = 200        # Number of iterations (Lower for a reasonable animation length)
# --------------------------------------------------------------------- #


def sphere_function(x: np.ndarray) -> float:
    """Objective function: Sphere function (minimize)."""
    return np.sum(x**2)


def levy_flight(Lambda: float = 1.5) -> np.ndarray:
    """
    Generate a Lévy flight step using Mantegna's algorithm.
    """
    # Ensure D is globally accessible or passed as a parameter
    global D
    sigma_u = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u, D)
    v = np.random.normal(0, 1, D)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step


def cuckoo_search_generator():
    """
    Generator version of Cuckoo Search.
    Yields the state of the nests at each generation.
    """
    global N, D, LB, UB, pa, alpha, MaxGen
    np.random.seed(42)

    # Step 1: Initialize nests randomly
    nests = np.random.uniform(LB, UB, (N, D))
    fitness = np.array([sphere_function(ind) for ind in nests])

    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_f = fitness[best_idx]

    # Yield the initial state
    yield {
        "nests": nests,
        "best": best,
        "best_f": best_f,
        "gen": 0
    }

    for gen in range(1, MaxGen + 1):
        
        # Step 2: Generate new solutions by Lévy flight (Global Search)
        for i in range(N):
            step = levy_flight()
            new_solution = nests[i] + alpha * step
            new_solution = np.clip(new_solution, LB, UB)
            new_f = sphere_function(new_solution)
            
            # Step 3: Greedy selection
            j = np.random.randint(0, N)
            if new_f < fitness[j]:
                nests[j] = new_solution
                fitness[j] = new_f

        # Step 4: Abandon worst nests and replace with local search
        n_abandon = int(pa * N)
        if n_abandon > 0:
            worst_indices = np.argsort(fitness)[-n_abandon:]
            
            for i in worst_indices:
                j, k = np.random.choice(np.delete(np.arange(N), i), 2, replace=False)
                r = np.random.rand() 
                new_local_solution = nests[i] + r * (nests[j] - nests[k])
                new_local_solution = np.clip(new_local_solution, LB, UB)
                
                new_local_f = sphere_function(new_local_solution)
                if new_local_f < fitness[i]:
                    nests[i] = new_local_solution
                    fitness[i] = new_local_f

        # Step 5: Update global best
        idx = np.argmin(fitness)
        if fitness[idx] < best_f:
            best_f = fitness[idx]
            best = nests[idx].copy()

        # Yield the state for this generation
        yield {
            "nests": nests,
            "best": best,
            "best_f": best_f,
            "gen": gen
        }

# --- Visualization Setup ---

# Create the generator object
# cs_generator = cuckoo_search_generator() # <-- REMOVED from here
cs_generator = None # <-- ADDED this line

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Initialize scatter plots (will be updated)
# Nests are blue dots
nests_scatter = ax.scatter([], [], c='blue', alpha=0.7, label='Nests')
# Best nest is a red star
best_scatter = ax.scatter([], [], c='red', marker='*', s=200, label='Best')

# Title for updating
title = ax.set_title("Cuckoo Search Initialization")

def setup_plot():
    """Sets up the background contour plot."""
    ax.set_xlim(LB, UB)
    ax.set_ylim(LB, UB)
    
    # Create a meshgrid for the contour plot
    x = np.linspace(LB, UB, 100)
    y = np.linspace(LB, UB, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z for the contour plot (Sphere function)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = sphere_function(np.array([X[i, j], Y[i, j]]))
            
    # Plot the filled contour
    ax.contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.6)
    
    # Add a colorbar
    plt.colorbar(ax.contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.6), ax=ax, label='Fitness Value (Cost)')
    
    ax.set_xlabel("Parameter 1 (x1)")
    ax.set_ylabel("Parameter 2 (x2)")
    ax.legend()
    ax.set_aspect('equal', 'box')
    return nests_scatter, best_scatter, title

def init_animation():
    """Initializes the animation frame."""
    global cs_generator # <-- ADDED this line
    cs_generator = cuckoo_search_generator() # <-- MOVED generator creation here

    setup_plot()
    nests_scatter.set_offsets(np.empty((0, 2)))
    best_scatter.set_offsets(np.empty((0, 2)))
    title.set_text("Generation 0")
    return nests_scatter, best_scatter, title

def update_animation(frame):
    """Updates the animation for each frame (generation)."""
    global cs_generator # <-- ADDED this line
    try:
        # Get the next state from the generator
        state = next(cs_generator)
        
        # Update nest positions (all N nests)
        nests_scatter.set_offsets(state["nests"])
        
        # Update best position (reshaped to (1, 2))
        best_scatter.set_offsets(state["best"].reshape(1, 2))
        
        # Update title
        title.set_text(f"Generation: {state['gen']} | Best Fitness: {state['best_f']:.2e}")
        
    except StopIteration:
        # Generator is exhausted
        pass
    
    return nests_scatter, best_scatter, title

# Create the animation
print("🚀 Starting Cuckoo Search animation...")
print("Please wait for the Matplotlib window to open.")

# Note: 'frames' specifies the total number of times to call update_animation
# We add 1 to MaxGen because we yield the initial state (gen 0)
ani = FuncAnimation(
    fig, 
    update_animation, 
    frames=MaxGen + 1, 
    init_func=init_animation,
    blit=True, 
    interval=300,  # Delay between frames in ms (Increased from 50)
    repeat=True   # Loop the animation (Changed from False)
)

plt.tight_layout()
plt.show()

print("\n✅ Animation finished.")