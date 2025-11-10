import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------- Objective Functions ---------------------- #
# All functions are designed to be minimized, with a global minimum of 0.

def sphere_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [0, 0, ..., 0]
    Bounds: [-10, 10]
    """
    return np.sum(x**2)


def rastrigin_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [0, 0, ..., 0]
    Bounds: [-5.12, 5.12]
    A_const = 10
    """
    A_const = 10
    return A_const * len(x) + np.sum(x**2 - A_const * np.cos(2 * math.pi * x))


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [1, 1, ..., 1]
    Bounds: [-2, 2]
    """
    # D=2 specific implementation for visualization
    x0 = x[0]
    x1 = x[1]
    return 100.0 * (x1 - x0**2.0)**2.0 + (1.0 - x0)**2.0


def ackley_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [0, 0, ..., 0]
    Bounds: [-5, 5] (Smaller for vis)
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2.0 * math.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20.0 + math.e


# ---------------------- Problem and Algorithm Config ---------------------- #
# --- CHOOSE YOUR PROBLEM TO VISUALIZE ---

# func_to_optimize = sphere_function
# LB, UB = -10, 10

func_to_optimize = rastrigin_function
LB, UB = -5.12, 5.12

# func_to_optimize = rosenbrock_function
# LB, UB = -2, 2

# func_to_optimize = ackley_function
# LB, UB = -5, 5

# ---------------------------------

N = 20              # Number of food sources (and employed/onlooker bees)
D = 2               # MUST BE 2 FOR 2D VISUALIZATION
MaxGen = 200        # Number of iterations
limit = 10          # Trial limit for scout bees
# ------------------------------------------------------------------------ #


def abc_algorithm_generator():
    """
    Generator version of Artificial Bee Colony (ABC).
    Yields the state of the food sources at each generation.
    """
    # Use the globally defined parameters
    global N, D, LB, UB, MaxGen, limit, func_to_optimize
    np.random.seed(42)

    # Step 1: Initialize food sources randomly
    foods = np.random.uniform(LB, UB, (N, D))
    fitness = np.array([func_to_optimize(ind) for ind in foods])
    
    # Trial counters for scout phase
    trial_counts = np.zeros(N, dtype=int)

    best_idx = np.argmin(fitness)
    best_food = foods[best_idx].copy()
    best_f = fitness[best_idx]

    # Yield the initial state
    yield {
        "foods": foods,
        "best_food": best_food,
        "best_f": best_f,
        "gen": 0
    }

    for gen in range(1, MaxGen + 1):
        
        # --- Employed Bee Phase ---
        for i in range(N):
            k_indices = np.delete(np.arange(N), i)
            k = np.random.choice(k_indices)
            
            phi = np.random.uniform(-1, 1, D)
            new_solution = foods[i] + phi * (foods[i] - foods[k])
            new_solution = np.clip(new_solution, LB, UB)
            
            new_f = func_to_optimize(new_solution)
            if new_f < fitness[i]:
                foods[i] = new_solution
                fitness[i] = new_f
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # --- Onlooker Bee Phase ---
        fit_vals = 1.0 / (1.0 + fitness + 1e-10) 
        prob = fit_vals / np.sum(fit_vals)
        
        for n in range(N):
            i = np.random.choice(N, p=prob)
            
            k_indices = np.delete(np.arange(N), i)
            k = np.random.choice(k_indices)
            
            phi = np.random.uniform(-1, 1, D)
            new_solution = foods[i] + phi * (foods[i] - foods[k])
            new_solution = np.clip(new_solution, LB, UB)
            
            new_f = func_to_optimize(new_solution)
            if new_f < fitness[i]:
                foods[i] = new_solution
                fitness[i] = new_f
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # --- Scout Bee Phase ---
        for i in range(N):
            if trial_counts[i] > limit:
                foods[i] = np.random.uniform(LB, UB, D)
                fitness[i] = func_to_optimize(foods[i])
                trial_counts[i] = 0

        # --- Update Global Best ---
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_f:
            best_f = fitness[current_best_idx]
            best_food = foods[current_best_idx].copy()

        # Yield the state for this generation
        yield {
            "foods": foods,
            "best_food": best_food,
            "best_f": best_f,
            "gen": gen
        }

# --- Visualization Setup ---

abc_generator = None
fig, ax = plt.subplots(figsize=(10, 8))

food_scatter = ax.scatter([], [], c='blue', alpha=0.7, label='Food Sources')
best_food_scatter = ax.scatter([], [], c='red', marker='*', s=200, label='Best Source')
title = ax.set_title("ABC Initialization")

def setup_plot():
    """Sets up the background contour plot."""
    global func_to_optimize, LB, UB, ax, fig
    ax.clear() 
    ax.set_xlim(LB, UB)
    ax.set_ylim(LB, UB)
    
    x = np.linspace(LB, UB, 100)
    y = np.linspace(LB, UB, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Use the selected objective function
            Z[i, j] = func_to_optimize(np.array([X[i, j], Y[i, j]]))
            
    # Use 20 levels for the contour plot
    levels = np.linspace(Z.min(), Z.max(), 20)
    if func_to_optimize == rosenbrock_function:
        # Rosenbrock has a huge range, log scale is better
        levels = np.logspace(np.log10(Z.min() + 1e-6), np.log10(Z.max()), 20)

    c_plot = ax.contourf(X, Y, Z, levels=levels, cmap='viridis_r', alpha=0.7)
    
    # Re-add the scatter objects to the cleared axis
    global food_scatter, best_food_scatter
    food_scatter = ax.scatter([], [], c='blue', alpha=0.7, label='Food Sources')
    best_food_scatter = ax.scatter([], [], c='red', marker='*', s=200, label='Best Source')
    
    ax.set_xlabel("Parameter 1 (x1)")
    ax.set_ylabel("Parameter 2 (x2)")
    ax.legend()
    ax.set_aspect('equal', 'box')
    # Set the title based on the function
    ax.set_title(f"ABC 2D Visualization: {func_to_optimize.__name__}")
    
    # Add a colorbar
    plt.colorbar(c_plot, ax=ax, label='Fitness Value (Cost)')

def init_animation():
    """Initializes the animation frame."""
    global abc_generator
    abc_generator = abc_algorithm_generator() # Create new generator
    
    setup_plot() # Setup the plot background
    
    global title
    title = ax.set_title(f"ABC 2D Visualization: {func_to_optimize.__name__}") # Reset title
    
    food_scatter.set_offsets(np.empty((0, 2)))
    best_food_scatter.set_offsets(np.empty((0, 2)))
    return food_scatter, best_food_scatter, title

def update_animation(frame):
    """Updates the animation for each frame (generation)."""
    global abc_generator, title
    try:
        state = next(abc_generator)
        
        food_scatter.set_offsets(state["foods"])
        best_food_scatter.set_offsets(state["best_food"].reshape(1, 2))
        title.set_text(f"Gen: {state['gen']} ({func_to_optimize.__name__}) | Best: {state['best_f']:.2e}")
        
    except StopIteration:
        pass # Generator is done, animation will loop
    
    return food_scatter, best_food_scatter, title

# Create the animation
print(f"🚀 Starting ABC animation for {func_to_optimize.__name__}...")
print("Please wait for the Matplotlib window to open.")

ani = FuncAnimation(
    fig, 
    update_animation, 
    frames=MaxGen + 1, 
    init_func=init_animation,
    blit=True, 
    interval=200,  # Slowed down
    repeat=True    # Repeats
)

plt.tight_layout()
plt.show()

print("\n✅ Animation finished.")