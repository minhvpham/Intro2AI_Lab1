# Algorithm Comparison Summary - Swarm Intelligence vs Traditional Search

## 📋 Project Overview

This project implements and compares **Swarm Intelligence** algorithms against **Traditional Search** algorithms for both continuous and discrete optimization problems.

## 🎯 Problems Addressed

1. **Continuous Optimization**: Rastrigin Function (highly multimodal)
2. **Discrete Optimization**: Traveling Salesman Problem (TSP)

## 🤖 Algorithms Implemented

### Swarm Intelligence Algorithms

| Algorithm | Problem Type | Location | Key Features |
|-----------|-------------|----------|--------------|
| **ACOR** | Continuous (Rastrigin) | `algo1 - ACO/rastrigin/` | Archive-based, Gaussian sampling |
| **PSO** | Continuous (Rastrigin) | `algo2 - PSO/continuous/` | Velocity-based, social/cognitive learning |
| **ACO** | Discrete (TSP) | `algo1 - ACO/tsp/` | Pheromone trails, probabilistic construction |
| **Hybrid PSO** | Discrete (TSP) | `algo2 - PSO/discrete/` | PSO + 2-opt local search |

### Traditional Search Algorithms

| Algorithm | Problem Type | Location | Key Features |
|-----------|-------------|----------|--------------|
| **Hill Climbing** | Continuous | `algo3 - Traditional/` | Steepest descent, random restarts, adaptive step |
| **Genetic Algorithm** | Continuous | `algo3 - Traditional/` | Real-valued encoding, BLX-α crossover |
| **Hill Climbing** | Discrete (TSP) | `algo3 - Traditional/` | 2-opt local search, random restarts |
| **A* Search** | Discrete (TSP) | `algo3 - Traditional/` | MST heuristic, optimal solution (small instances) |
| **Genetic Algorithm** | Discrete (TSP) | `algo3 - Traditional/` | Order crossover, swap/inversion mutation |

## 📂 Project Structure

```
Intro2AI_Lab1/
│
├── algo1 - ACO/
│   ├── rastrigin/
│   │   ├── rastrigin.py          # ACOR implementation
│   │   ├── benchmark_functions.py
│   │   └── visualization.py
│   └── tsp/
│       ├── ACO.py                # ACO for TSP
│       └── gui_main.py
│
├── algo2 - PSO/
│   ├── continuous/
│   │   ├── pso.py                # PSO for Rastrigin
│   │   └── main.py
│   └── discrete/
│       └── pso_tsp.py            # Hybrid PSO for TSP
│
└── algo3 - Traditional/          ⭐ NEW!
    ├── continuous_traditional.py  # Hill Climbing + GA (continuous)
    ├── tsp_traditional.py         # HC + A* + GA (TSP)
    ├── test_all.py                # Quick test script
    └── README.md                  # Detailed documentation
├── compare_all_algorithms.py  # Comprehensive comparison
```

## 🚀 How to Run

### Option 1: Test Individual Algorithm Types

```bash
# Test continuous optimization algorithms
cd "algo3 - Traditional"
python continuous_traditional.py

# Test TSP algorithms
python tsp_traditional.py
```

### Option 2: Quick Test All Implementations

```bash
cd "algo3 - Traditional"
python test_all.py
```

### Option 3: Full Comparison (Recommended!)

```bash
cd "algo3 - Traditional"
python compare_all_algorithms.py
```

This will:
- Run all algorithms on both problems
- Generate comparison plots
- Save results as PNG files
- Display comprehensive summary

## 📊 Expected Outputs

### Console Output
- Real-time progress for each algorithm
- Best solution found by each algorithm
- Time elapsed
- Final comparison table

### Generated Plots
1. **`comparison_rastrigin_convergence.png`** - Convergence curves for Rastrigin
2. **`comparison_rastrigin_bars.png`** - Cost and time bar charts for Rastrigin
3. **`comparison_tsp_convergence.png`** - Convergence curves for TSP
4. **`comparison_tsp_bars.png`** - Cost and time bar charts for TSP
5. **`comparison_tsp_solutions.png`** - Visual comparison of TSP solutions

## 🔍 Algorithm Details

### Continuous Optimization (Rastrigin)

#### Hill Climbing
- **Strategy**: Steepest descent with adaptive step size
- **Strengths**: Fast, simple, good for smooth landscapes
- **Weaknesses**: Can get stuck in local minima
- **Solution**: Multiple random restarts (10 restarts default)

#### Genetic Algorithm
- **Encoding**: Real-valued vectors
- **Selection**: Tournament selection (size=3)
- **Crossover**: BLX-α (Blend Crossover) at 80% rate
- **Mutation**: Gaussian mutation at 10% rate
- **Strengths**: Good balance of exploration/exploitation
- **Parameters**: Pop=50, Gen=100

#### ACOR (Swarm)
- **Strategy**: Archive of solutions + Gaussian sampling
- **Selection**: Probabilistic based on solution quality
- **Strengths**: Effective for multimodal functions
- **Parameters**: Archive=30, Samples=50, Iter=100

#### PSO (Swarm)
- **Strategy**: Particle velocities guided by personal and global best
- **Strengths**: Fast convergence, simple implementation
- **Parameters**: Particles=30, w=0.8, c1=c2=0.5, Iter=100

### Discrete Optimization (TSP)

#### Hill Climbing (2-opt)
- **Strategy**: Iteratively reverse tour segments that reduce cost
- **Improvement**: First improvement (fast)
- **Restarts**: 10 random starting tours
- **Strengths**: Fast, simple, reliable local optimization

#### A* Search
- **Heuristic**: Minimum Spanning Tree (MST) lower bound
- **Completeness**: Guaranteed optimal (if completes)
- **Limitation**: Only practical for ≤15 cities
- **Complexity**: Exponential in worst case

#### Genetic Algorithm
- **Encoding**: Permutation of city indices
- **Crossover**: Order Crossover (OX) at 90% rate
- **Mutation**: Swap + Inversion at 20% rate
- **Strengths**: Preserves good sub-tours
- **Parameters**: Pop=100, Gen=200

#### ACO (Swarm)
- **Strategy**: Pheromone trails guide probabilistic tour construction
- **Update**: Evaporation + deposition based on tour quality
- **Strengths**: Emergent optimization through stigmergy
- **Parameters**: Ants=20, Iter=100, α=1.0, β=2.0, ρ=0.5

#### Hybrid PSO (Swarm)
- **Strategy**: PSO velocity update + 2-opt local search
- **Velocity**: Discrete operations (swap sequences)
- **Strengths**: Global search + local refinement
- **Parameters**: Particles=20, Iter=100

## 📈 Performance Comparison

### Rastrigin Function (5D)

**Typical Results:**
```
Algorithm              Best Cost      Time (s)
-------------------------------------------------
Hill Climbing          ~2-10          ~0.5-1.0
Genetic Algorithm      ~1-5           ~1.0-2.0
ACOR (Swarm)          ~0.5-3         ~0.8-1.5
PSO (Swarm)           ~0.5-2         ~0.3-0.8
```

**Winner**: Usually PSO or ACOR (swarm intelligence excels at multimodal functions)

### TSP (20 Cities)

**Typical Results:**
```
Algorithm              Best Cost      Time (s)    Notes
----------------------------------------------------------------
Hill Climbing          ~380-420       ~0.2-0.5    Fast, decent
A* Search             N/A            N/A         Too many cities
Genetic Algorithm      ~360-400       ~2-5        Good quality
ACO (Swarm)           ~350-390       ~15-25      Good but slow
Hybrid PSO (Swarm)    ~340-380       ~3-8        Often best
```

**Winner**: Usually Hybrid PSO (combines global search with local optimization)

## 🎓 Key Insights

### When to Use Swarm Intelligence
✅ **Highly multimodal functions** (many local minima)  
✅ **Complex, non-convex landscapes**  
✅ **Need robust solutions** (less sensitive to initialization)  
✅ **Willing to trade time for quality**  

### When to Use Traditional Algorithms
✅ **Simple, smooth landscapes** (Hill Climbing)  
✅ **Need guaranteed optimal** (A* for small instances)  
✅ **Fast solutions needed** (Hill Climbing with restarts)  
✅ **Well-understood problem structure** (GA with domain-specific operators)  

### Hybrid Approaches
🌟 **Best of Both Worlds**: Combine swarm intelligence for global search with traditional local search (e.g., Hybrid PSO = PSO + 2-opt)

## 🛠️ Customization Guide

### Adjust Problem Difficulty

**Rastrigin:**
```python
# In compare_all_algorithms.py
run_continuous_comparison(
    n_dims=10,              # Increase dimensions (harder)
    bounds=(-5.12, 5.12),   # Standard bounds
    function=rastrigin
)
```

**TSP:**
```python
# In compare_all_algorithms.py
run_tsp_comparison(
    n_cities=30,            # More cities (harder)
    map_size=100,
    seed=42                 # Change seed for different instances
)
```

### Tune Algorithm Parameters

See detailed parameter guides in `README.md`

## 📚 References

1. **Rastrigin Function**: Standard benchmark for multimodal optimization
2. **TSP**: Classic NP-hard combinatorial optimization problem
3. **Swarm Intelligence**: Dorigo & Stützle (2004), Kennedy & Eberhart (1995)
4. **Traditional Search**: Russell & Norvig (2020), Goldberg (1989)

## ⚠️ Important Notes

1. **A* Limitation**: Only use for TSP with ≤15 cities (computational complexity)
2. **Random Seeds**: Results vary between runs; use seeds for reproducibility
3. **Convergence**: Swarm algorithms may need more iterations for convergence
4. **Visualization**: Log scale used for convergence plots (better visualization)

## 🎯 Conclusion

This implementation provides a comprehensive comparison framework showing:
- **Swarm Intelligence** excels at complex, multimodal problems
- **Traditional Algorithms** are faster and simpler for well-behaved problems
- **Hybrid Approaches** often provide the best performance
- **No single algorithm dominates** - choice depends on problem characteristics

## 👨‍💻 Usage for Assignment

To fulfill your assignment requirements:

1. ✅ **3+ Traditional Algorithms Implemented**:
   - Hill Climbing (continuous + discrete)
   - Genetic Algorithm (continuous + discrete)
   - A* Search (discrete)

2. ✅ **Comparison with Swarm Intelligence**:
   - ACOR & PSO (continuous)
   - ACO & Hybrid PSO (discrete)

3. ✅ **Both Problem Types**:
   - Continuous: Rastrigin function
   - Discrete: TSP

4. ✅ **Comprehensive Analysis**:
   - Convergence plots
   - Performance comparison
   - Time analysis
   - Visual results

**Simply run `compare_all_algorithms.py` to generate all required results!**

---

**Good luck with your assignment! 🚀**
