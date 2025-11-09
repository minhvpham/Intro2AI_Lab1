# Traditional Search Algorithms vs Swarm Intelligence

This directory contains implementations of traditional search algorithms and a comprehensive comparison framework with swarm intelligence algorithms.

## 📁 Files

- **`continuous_traditional.py`** - Hill Climbing & Genetic Algorithm for continuous optimization (Rastrigin, Sphere, etc.)
- **`tsp_traditional.py`** - Hill Climbing, A* Search, & Genetic Algorithm for TSP
- **`compare_all_algorithms.py`** - Comprehensive comparison framework

## 🔍 Algorithms Implemented

### Traditional Algorithms

#### For Continuous Optimization (Rastrigin Function)
1. **Hill Climbing (Steepest Ascent/Descent)**
   - Uses adaptive step size
   - Random restarts to escape local minima
   - Explores neighborhood in all dimensions

2. **Genetic Algorithm**
   - Real-valued encoding
   - BLX-alpha crossover (Blend Crossover)
   - Gaussian mutation
   - Tournament selection
   - Elitism

#### For Discrete Optimization (TSP)
1. **Hill Climbing**
   - 2-opt local search
   - First improvement strategy
   - Multiple random restarts

2. **A* Search**
   - MST-based admissible heuristic
   - Only practical for small instances (≤15 cities)
   - Guaranteed optimal solution (if completed)

3. **Genetic Algorithm**
   - Permutation encoding
   - Order Crossover (OX)
   - Swap and inversion mutation
   - Tournament selection
   - Elitism

### Swarm Intelligence (Existing)
- **ACOR** (Ant Colony Optimization for Continuous) - Rastrigin
- **PSO** (Particle Swarm Optimization) - Rastrigin
- **ACO** (Ant Colony Optimization) - TSP
- **Hybrid PSO** (PSO + 2-opt) - TSP

## 🚀 Usage

### Run Individual Algorithms

#### Continuous Optimization
```bash
python continuous_traditional.py
```
This will:
- Run Hill Climbing and GA on 5D Rastrigin function
- Display convergence comparison plots
- Show results summary

#### TSP Optimization
```bash
python tsp_traditional.py
```
This will:
- Run Hill Climbing, A* (if n_cities ≤ 15), and GA on TSP
- Visualize each solution
- Display convergence comparison
- Show results summary

### Run Comprehensive Comparison

```bash
python compare_all_algorithms.py
```

This will:
1. **Compare all algorithms on Rastrigin function (5D)**
   - Hill Climbing vs GA vs ACOR vs PSO
   
2. **Compare all algorithms on TSP (20 cities)**
   - Hill Climbing vs A* vs GA vs ACO vs Hybrid PSO

3. **Generate comparison plots:**
   - Convergence curves (log scale)
   - Performance bar charts (cost & time)
   - TSP solution visualizations

4. **Save results:**
   - `comparison_rastrigin_convergence.png`
   - `comparison_rastrigin_bars.png`
   - `comparison_tsp_convergence.png`
   - `comparison_tsp_bars.png`
   - `comparison_tsp_solutions.png`

## 📊 Customization

### Continuous Optimization Parameters

**Hill Climbing:**
```python
hc = HillClimbing(
    cost_function=rastrigin,
    n_dims=5,              # Problem dimensions
    bounds=[-5.12, 5.12],  # Search space
    step_size=0.5,         # Initial step size
    max_iterations=200,    # Max iterations per restart
    n_restarts=10,         # Number of restarts
    adaptive_step=True     # Enable adaptive step sizing
)
```

**Genetic Algorithm:**
```python
ga = GeneticAlgorithm(
    cost_function=rastrigin,
    n_dims=5,
    bounds=[-5.12, 5.12],
    population_size=50,     # Population size
    n_generations=100,      # Number of generations
    crossover_rate=0.8,     # Crossover probability
    mutation_rate=0.1,      # Mutation probability
    elite_size=2,           # Number of elites
    tournament_size=3       # Tournament size
)
```

### TSP Parameters

**Hill Climbing:**
```python
hc_tsp = HillClimbingTSP(
    cities=cities,
    n_restarts=10,          # Number of random restarts
    max_iterations=1000     # Max iterations per restart
)
```

**A* Search:**
```python
astar_tsp = AStarTSP(
    cities=cities,
    time_limit=60           # Time limit in seconds
)
```
⚠️ **Note:** A* is only practical for ≤15 cities

**Genetic Algorithm:**
```python
ga_tsp = GeneticAlgorithmTSP(
    cities=cities,
    population_size=100,
    n_generations=200,
    crossover_rate=0.9,
    mutation_rate=0.2,
    elite_size=5,
    tournament_size=5
)
```

### Comparison Framework

Edit parameters in `compare_all_algorithms.py`:

```python
# Continuous optimization
continuous_results = run_continuous_comparison(
    n_dims=5,                    # Problem dimensions
    bounds=(-5.12, 5.12),        # Search bounds
    function=rastrigin,          # Objective function
    function_name="Rastrigin"    # Display name
)

# TSP
tsp_results, cities = run_tsp_comparison(
    n_cities=20,     # Number of cities
    map_size=100,    # Map dimensions
    seed=42          # Random seed
)
```

## 📈 Expected Results

### Rastrigin Function (5D)
- **Swarm Intelligence** typically finds better solutions for highly multimodal functions
- **GA** balances exploration and exploitation well
- **Hill Climbing** may get stuck in local minima despite restarts

### TSP
- **A\*** finds optimal solution (for small instances)
- **Hybrid PSO** (swarm) often performs best due to 2-opt local search
- **GA** provides good quality solutions with reasonable time
- **Hill Climbing** fast but quality depends on restarts

## 🔧 Dependencies

```bash
pip install numpy matplotlib
```

## 📝 Notes

1. **A* Limitations:** Only use A* for TSP instances with ≤15 cities. For larger instances, it will automatically return a greedy solution.

2. **Random Seeds:** For reproducible results, set `np.random.seed()` at the beginning of your script.

3. **Convergence Plots:** Use log scale (`plt.yscale('log')`) for better visualization of convergence curves.

4. **Performance:** Swarm algorithms generally require more function evaluations but often find better solutions for complex landscapes.

## 🎯 Key Insights

### Continuous Optimization
- **ACOR & PSO** excel at escaping local minima in multimodal functions
- **GA** provides consistent performance with proper tuning
- **Hill Climbing** is fast but heavily dependent on starting points

### TSP
- **A\*** guarantees optimality but infeasible for large instances
- **Swarm algorithms** (ACO, Hybrid PSO) leverage collective intelligence
- **GA with OX crossover** preserves good sub-tours effectively
- **Hill Climbing with 2-opt** provides fast local optimization

## 📚 References

1. **Hill Climbing:** Russell & Norvig - "Artificial Intelligence: A Modern Approach"
2. **Genetic Algorithms:** Goldberg - "Genetic Algorithms in Search, Optimization, and Machine Learning"
3. **A\* Search:** Hart, Nilsson, Raphael - "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
4. **Swarm Intelligence:** Dorigo & Stützle - "Ant Colony Optimization"; Kennedy & Eberhart - "Particle Swarm Optimization"

---

**Author:** AI Lab 1 - Intro to AI Course  
**Date:** November 2025
