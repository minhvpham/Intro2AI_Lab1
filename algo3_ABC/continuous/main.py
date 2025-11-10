import numpy as np
import math

# ---------------------- Objective Functions ---------------------- #
# All functions are designed to be minimized, with a global minimum of 0.

def sphere_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [0, 0, ..., 0]
    Bounds: Typically [-100, 100]
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
    Bounds: Typically [-10, 10] or [-2.048, 2.048]
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)


def ackley_function(x: np.ndarray) -> float:
    """
    Global minimum: 0 at x = [0, 0, ..., 0]
    Bounds: [-32.768, 32.768]
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2.0 * math.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20.0 + math.e


# ---------------------- Problem and Algorithm Config ---------------------- #
# --- CHOOSE YOUR PROBLEM HERE ---
#
# func_to_optimize = sphere_function
# LB, UB = -100, 100
#
func_to_optimize = rastrigin_function
LB, UB = -5.12, 5.12
#
# func_to_optimize = rosenbrock_function
# LB, UB = -10, 10
#
# func_to_optimize = ackley_function
# LB, UB = -32.768, 32.768
#
# ---------------------------------

D = 30              # Dimension of the problem
N = 50              # Number of food sources (population size)
MaxGen = 1000       # Number of iterations
limit = 20          # Trial limit for scout bees
# ------------------------------------------------------------------------ #


def artificial_bee_colony(func_to_optimize, LB, UB, D, N, MaxGen, limit):
    """Main ABC algorithm for continuous function minimization.
    
    Returns:
        tuple: (best_food_vector, best_fitness, history_of_best_fitness)
    """
    
    np.random.seed(42)

    # Step 1: Initialize food sources randomly within bounds
    foods = np.random.uniform(LB, UB, (N, D))
    fitness = np.array([func_to_optimize(ind) for ind in foods])
    
    # Trial counters for scout phase
    trial_counts = np.zeros(N, dtype=int)

    # Find and store the initial best
    best_idx = np.argmin(fitness)
    best_food = foods[best_idx].copy()
    best_f = fitness[best_idx]
    
    # Store history of best fitness
    history_list = [best_f]
    
    print(f"Initial Best Fitness ({func_to_optimize.__name__}): {best_f:.6e}")

    for gen in range(1, MaxGen + 1):
        
        # --- Employed Bee Phase ---
        for i in range(N):
            # Pick a random partner 'k' (different from 'i')
            k = np.random.choice(np.delete(np.arange(N), i))
            
            # Generate a new solution v_i
            phi = np.random.uniform(-1, 1, D)
            new_solution = foods[i] + phi * (foods[i] - foods[k])
            new_solution = np.clip(new_solution, LB, UB)
            
            # Greedy selection (for minimization)
            new_f = func_to_optimize(new_solution)
            if new_f < fitness[i]:
                foods[i] = new_solution
                fitness[i] = new_f
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # --- Onlooker Bee Phase ---
        
        # Calculate probabilities for roulette wheel selection
        # For minimization, better fitness is smaller.
        # We convert fitness to a "quality" value (higher is better)
        # 1 / (1 + fitness) handles 0 fitness and preserves order
        fit_vals = 1.0 / (1.0 + fitness + 1e-10) 
        prob = fit_vals / np.sum(fit_vals)
        
        for n in range(N):
            # Select source 'i' based on probability
            i = np.random.choice(N, p=prob)
            
            # Pick a random partner 'k' (different from 'i')
            k = np.random.choice(np.delete(np.arange(N), i))
            
            # Generate a new solution v_i
            phi = np.random.uniform(-1, 1, D)
            new_solution = foods[i] + phi * (foods[i] - foods[k])
            new_solution = np.clip(new_solution, LB, UB)
            
            # Greedy selection
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
                # Abandon this source and replace with a new random solution
                foods[i] = np.random.uniform(LB, UB, D)
                fitness[i] = func_to_optimize(foods[i])
                trial_counts[i] = 0

        # --- Update Global Best ---
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_f:
            best_f = fitness[current_best_idx]
            best_food = foods[current_best_idx].copy()

        # Log progress
        if gen % 100 == 0:
            print(f"Gen {gen}: Best Fitness = {best_f:.6e}")
            
        history_list.append(best_f)
            
    return best_food, best_f, history_list


# ----------------------------- Run Algorithm ----------------------------- #
if __name__ == "__main__":
    print(f"--- Running ABC for: {func_to_optimize.__name__} ---")
    print(f"Bounds: [{LB}, {UB}], Dimensions: {D}, Population: {N}")
    
    final_solution, final_fitness, _ = artificial_bee_colony(
        func_to_optimize, LB, UB, D, N, MaxGen, limit
    )
    
    print("\n--- ✅ ABC Algorithm Finished ---")
    print(f"Final Best Fitness: {final_fitness:.6e}")
    
    # For Rosenbrock, the optimal is [1, 1, ...], for others it's [0, 0, ...]
    if func_to_optimize == rosenbrock_function:
        print("Note: Rosenbrock's global minimum is at x = [1, 1, ..., 1]")
    else:
        print("Note: The global minimum for this function is at x = [0, 0, ..., 0]")
    
    # print("\nFinal Best Solution Vector (first 5 dims):")
    # print(final_solution[:5])