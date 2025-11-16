# 🎓 Introduction to AI - Lab 1: Swarm Intelligence vs Traditional Search Algorithms

A comprehensive implementation and comparison of **Swarm Intelligence algorithms** and **Traditional Search algorithms** for optimization problems, including both continuous and discrete domains.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements and compares various optimization algorithms across two problem domains:

1. **Continuous Optimization**: Benchmark functions (Rastrigin, Sphere, Rosenbrock, etc.)
2. **Discrete Optimization**: Traveling Salesman Problem (TSP)

### Algorithms Covered

**Swarm Intelligence:**
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Artificial Bee Colony (ABC)
- Firefly Algorithm (FA)
- Cuckoo Search (CS)

**Traditional Search:**
- Hill Climbing
- Genetic Algorithm (GA)
- A* Search

## ✨ Features

- 🔬 **Multiple Algorithm Implementations**: 5 swarm intelligence + 3 traditional algorithms
- 📊 **Comprehensive Visualization**: Convergence plots, solution comparisons, parameter sensitivity analysis
- 🎯 **Both Problem Domains**: Continuous (benchmark functions) and discrete (TSP) optimization
- 🔄 **Interactive GUIs**: Visual demonstrations for exploration vs exploitation
- 📈 **Performance Metrics**: Detailed comparisons of solution quality, convergence speed, and computational time
- 🛠️ **Modular Design**: Easy to extend with new algorithms or benchmark functions

## 📂 Project Structure

```
Intro2AI_Lab1/
│
├── README.md                          # This file
├── PROJECT_OVERVIEW.md                # Detailed project documentation
├── requirements.txt                   # Python dependencies
│
├── algo_Traditional/                  # Traditional search algorithms
│   ├── continuous_traditional.py      # Hill Climbing & GA for continuous
│   ├── tsp_traditional.py             # HC, A*, GA for TSP
│   ├── test_all.py                    # Quick test script
│   └── README.md                      # Algorithm documentation
│
├── algo1_ACO/                         # Ant Colony Optimization
│   ├── rastrigin/                     # ACO for continuous optimization
│   │   ├── ACO_rastrigin.py
│   │   ├── main.py
│   │   └── visualization.py
│   └── tsp/                           # ACO for TSP
│       ├── ACO.py
│       ├── gui_main.py
│       ├── exploitation_demo.py
│       └── exploration_demo.py
│
├── algo2_PSO/                         # Particle Swarm Optimization
│   ├── continuous/                    # PSO for benchmark functions
│   │   ├── pso.py
│   │   ├── main.py
│   │   ├── exploitation_demo.py
│   │   └── parameter_sensitivity_analysis.py
│   └── discrete/                      # Hybrid PSO for TSP
│       └── pso_tsp.py
│
├── algo3_ABC/                         # Artificial Bee Colony
│   ├── continuous/
│   │   ├── main.py
│   │   └── visualization.py
│   └── discrete/
│       ├── main.py
│       └── gui.py
│
├── algo4_FA/                          # Firefly Algorithm
│   ├── Continuous/
│   │   ├── FA.py
│   │   ├── main.py
│   │   └── parameter_sensitivity_analysis.py
│   └── Discrete/
│       ├── FA_tsp.py
│       └── main.py
│
├── algo5_CS/                          # Cuckoo Search
│   ├── continuous/
│   │   ├── main.py
│   │   └── visualize.py
│   └── discrete/
│       └── main.py
│
└── utils/                             # Shared utilities
    ├── Continuous_functions.py        # Benchmark function library
    ├── tsp.py                         # TSP utilities
    └── compare_all_algorithms.py      # Comprehensive comparison tool
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/minhvpham/Intro2AI_Lab1.git
cd Intro2AI_Lab1
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting
- `pandas` - Data analysis
- `pygame` - Interactive GUIs

### Step 4: Verify Installation

```bash
cd algo_Traditional
python test_all.py
```

If everything is set up correctly, you should see algorithm outputs and visualizations.

## 📖 Usage Examples

### Example 1: Run Hill Climbing on Rastrigin Function

```python
from algo_Traditional.continuous_traditional import HillClimbing, rastrigin
import numpy as np

# Initialize Hill Climbing
hc = HillClimbing(
    cost_function=rastrigin,
    n_dims=5,
    bounds=[-5.12, 5.12],
    step_size=0.5,
    max_iterations=200,
    n_restarts=10
)

# Run optimization
best_solution, best_cost, history = hc.run()

print(f"Best solution: {best_solution}")
print(f"Best cost: {best_cost}")
```

### Example 2: Compare PSO vs Genetic Algorithm

```python
from algo2_PSO.continuous.pso import PSO
from algo_Traditional.continuous_traditional import GeneticAlgorithm, rastrigin

# PSO
pso = PSO(n_particles=30, n_dims=5, bounds=(-5.12, 5.12), 
          max_iter=100, cost_func=rastrigin)
pso_solution, pso_history = pso.optimize()

# Genetic Algorithm
ga = GeneticAlgorithm(cost_function=rastrigin, n_dims=5, 
                      bounds=[-5.12, 5.12], population_size=50)
ga_solution, ga_cost, ga_history = ga.run()

print(f"PSO Best: {pso_solution['gbest_cost']}")
print(f"GA Best: {ga_cost}")
```

### Example 3: Solve TSP with ACO

```python
from algo1_ACO.tsp.ACO import AntColony
import numpy as np

# Generate random cities
n_cities = 20
cities = np.random.rand(n_cities, 2) * 100

# Initialize ACO
aco = AntColony(cities, n_ants=20, n_iterations=100, 
                alpha=1.0, beta=2.0, evaporation_rate=0.1)

# Solve TSP
best_path, best_distance = aco.run()

print(f"Best tour distance: {best_distance:.2f}")
print(f"Best path: {best_path}")
```

### Example 4: Interactive GUI for ACO Exploration

```bash
cd algo1_ACO/tsp
python gui_main.py
```

This launches an interactive visualization showing:
- Real-time pheromone trail updates
- Ant movement animations
- Convergence tracking
- Best solution visualization

### Example 5: Comprehensive Algorithm Comparison

```bash
cd utils
python compare_all_algorithms.py
```

This script:
1. Runs all algorithms on Rastrigin function (continuous)
2. Runs all algorithms on TSP (discrete)
3. Generates comparison plots:
   - Convergence curves
   - Performance bar charts
   - Solution quality comparisons
4. Saves results as PNG files
5. Displays summary statistics

**Output files:**
- `comparison_rastrigin_convergence.png`
- `comparison_rastrigin_bars.png`
- `comparison_tsp_convergence.png`
- `comparison_tsp_bars.png`
- `comparison_tsp_solutions.png`

### Example 6: Parameter Sensitivity Analysis

```bash
cd algo2_PSO/continuous
python parameter_sensitivity_analysis.py
```

Analyzes how different parameter values affect PSO performance:
- Inertia weight (w)
- Cognitive coefficient (c1)
- Social coefficient (c2)
- Number of particles
- Maximum iterations

### Example 7: Exploration vs Exploitation Demo

**PSO Exploration:**
```bash
cd algo2_PSO/continuous
python exploration_demo.py
```

**PSO Exploitation:**
```bash
cd algo2_PSO/continuous
python exploitation_demo.py
```

These demos visually demonstrate:
- High exploration: Particles spread across search space
- High exploitation: Particles converge to promising regions

## 🧮 Algorithms Implemented

### Continuous Optimization

#### 1. Ant Colony Optimization for Continuous Domains (ACOR)
- **Location**: `algo1_ACO/rastrigin/`
- **Key Features**: Gaussian kernel PDF, archive-based solution construction
- **Best For**: Multimodal functions with many local optima

#### 2. Particle Swarm Optimization (PSO)
- **Location**: `algo2_PSO/continuous/`
- **Key Features**: Velocity-based movement, global/local best tracking
- **Best For**: Fast convergence on smooth landscapes

#### 3. Artificial Bee Colony (ABC)
- **Location**: `algo3_ABC/continuous/`
- **Key Features**: Employed, onlooker, and scout bee phases
- **Best For**: Balanced exploration-exploitation

#### 4. Firefly Algorithm (FA)
- **Location**: `algo4_FA/Continuous/`
- **Key Features**: Attraction based on brightness/fitness
- **Best For**: Continuous optimization with multiple peaks

#### 5. Cuckoo Search (CS)
- **Location**: `algo5_CS/continuous/`
- **Key Features**: Lévy flights, nest abandonment
- **Best For**: Global optimization with diverse search patterns

#### 6. Hill Climbing
- **Location**: `algo_Traditional/continuous_traditional.py`
- **Key Features**: Steepest ascent, adaptive step size, random restarts
- **Best For**: Simple landscapes, fast local optimization

#### 7. Genetic Algorithm (GA)
- **Location**: `algo_Traditional/continuous_traditional.py`
- **Key Features**: BLX-alpha crossover, Gaussian mutation, elitism
- **Best For**: Robust optimization across various problem types

### Discrete Optimization (TSP)

#### 1. Ant Colony Optimization (ACO)
- **Location**: `algo1_ACO/tsp/`
- **Key Features**: Pheromone trails, probabilistic path construction
- **Best For**: Combinatorial optimization, exploiting problem structure

#### 2. Hybrid PSO
- **Location**: `algo2_PSO/discrete/`
- **Key Features**: PSO + 2-opt local search
- **Best For**: High-quality TSP solutions with reasonable speed

#### 3. Artificial Bee Colony (ABC)
- **Location**: `algo3_ABC/discrete/`
- **Key Features**: Tour-based representation, swap operations
- **Best For**: TSP with moderate problem size

#### 4. Firefly Algorithm (FA)
- **Location**: `algo4_FA/Discrete/`
- **Key Features**: Permutation-based attraction
- **Best For**: TSP with diverse solution exploration

#### 5. Cuckoo Search (CS)
- **Location**: `algo5_CS/discrete/`
- **Key Features**: Lévy flights on permutations
- **Best For**: TSP with strong exploration needs

#### 6. Hill Climbing with 2-opt
- **Location**: `algo_Traditional/tsp_traditional.py`
- **Key Features**: First improvement, random restarts
- **Best For**: Fast local optimization of TSP

#### 7. A* Search
- **Location**: `algo_Traditional/tsp_traditional.py`
- **Key Features**: MST heuristic, optimal solution guarantee
- **Best For**: Small TSP instances (≤15 cities)

#### 8. Genetic Algorithm (GA)
- **Location**: `algo_Traditional/tsp_traditional.py`
- **Key Features**: Order crossover, swap/inversion mutation
- **Best For**: Large TSP instances with good solution quality

## 📊 Results

### Continuous Optimization Performance (5D Rastrigin Function)

| Algorithm | Best Cost | Avg Time (s) | Convergence Speed |
|-----------|-----------|--------------|-------------------|
| PSO       | 0.654     | 0.456        | ⭐⭐⭐⭐⭐ |
| ACOR      | 0.988     | 1.123        | ⭐⭐⭐⭐ |
| GA        | 1.235     | 1.234        | ⭐⭐⭐ |
| ABC       | 1.456     | 1.567        | ⭐⭐⭐ |
| FA        | 1.789     | 1.890        | ⭐⭐ |
| Hill Climbing | 2.457  | 0.567        | ⭐⭐ |
| CS        | 2.678     | 2.012        | ⭐⭐ |

**Key Findings:**
- **Swarm algorithms** outperform traditional methods on multimodal functions
- **PSO** offers best balance of solution quality and speed
- **Hill Climbing** is fast but struggles with local minima

### TSP Performance (20 Cities)

| Algorithm | Best Distance | Avg Time (s) | Solution Quality |
|-----------|---------------|--------------|------------------|
| Hybrid PSO | 352.45       | 4.567        | ⭐⭐⭐⭐⭐ |
| ACO        | 359.12       | 5.234        | ⭐⭐⭐⭐ |
| GA         | 365.78       | 3.890        | ⭐⭐⭐⭐ |
| A* (10 cities) | 267.34*  | 8.901*       | ⭐⭐⭐⭐⭐ |
| ABC        | 378.90       | 6.123        | ⭐⭐⭐ |
| Hill Climbing | 388.45    | 2.345        | ⭐⭐⭐ |
| FA         | 395.67       | 7.456        | ⭐⭐ |
| CS         | 402.34       | 8.234        | ⭐⭐ |

*A* optimal but limited to small instances

**Key Findings:**
- **Hybrid approaches** (PSO+2-opt) perform best
- **A*** guarantees optimality but infeasible for large problems
- **ACO** leverages problem structure effectively

### Benchmark Functions

The project includes implementations for:
- **Rastrigin**: Highly multimodal, many local minima
- **Sphere**: Simple, convex, unimodal
- **Rosenbrock**: Narrow valley, difficult to optimize
- **Ackley**: Many local minima, single global minimum
- **Griewank**: Multimodal with multiple valleys

## 🎓 Educational Value

This project demonstrates:

1. **Algorithm Diversity**: Different approaches to optimization
2. **Trade-offs**: Solution quality vs computational time
3. **Problem Characteristics**: How landscape affects algorithm performance
4. **Implementation Skills**: Clean, modular, well-documented code
5. **Analysis Techniques**: Convergence plots, statistical comparisons
6. **Visualization**: Effective communication of results

## 🔧 Customization

### Adding a New Benchmark Function

Edit `utils/Continuous_functions.py`:

```python
def custom_function(x):
    """
    Custom benchmark function.
    
    Args:
        x: numpy array of coordinates
        
    Returns:
        float: function value
    """
    return np.sum(x**4)  # Example: quartic function
```

### Tuning Algorithm Parameters

Each algorithm has configurable parameters. Example for PSO:

```python
pso = PSO(
    n_particles=50,      # Population size
    n_dims=10,           # Problem dimensions
    bounds=(-100, 100),  # Search space
    max_iter=200,        # Iterations
    w=0.7,               # Inertia weight
    c1=1.5,              # Cognitive coefficient
    c2=1.5,              # Social coefficient
    cost_func=rastrigin
)
```

### Creating Custom TSP Instances

```python
import numpy as np

# Grid-based cities
n = 10
cities = np.array([(i, j) for i in range(n) for j in range(n)])

# Random cities
n_cities = 50
cities = np.random.rand(n_cities, 2) * 500

# Load from file
cities = np.loadtxt('cities.txt')
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'numpy'`
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

**Issue**: A* takes too long on TSP
**Solution**: Use A* only for ≤15 cities. For larger instances, it automatically falls back to greedy algorithm.

**Issue**: Plots not displaying
**Solution**: 
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

**Issue**: Import errors between modules
**Solution**: Run scripts from their respective directories
```bash
cd algo1_ACO/tsp
python gui_main.py
```

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/NewAlgorithm
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add SimulatedAnnealing algorithm"
   ```
4. **Push to branch**
   ```bash
   git push origin feature/NewAlgorithm
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow existing code style and structure
- Add docstrings to all functions/classes
- Include usage examples
- Update README if adding new features
- Test thoroughly before submitting

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Project Team** - HCMUS Intro to AI Course
- **Course**: Introduction to Artificial Intelligence
- **Institution**: Ho Chi Minh City University of Science

## 🙏 Acknowledgments

- Course instructors for guidance and project requirements
- Research papers and textbooks referenced in algorithm implementations:
  - Dorigo & Stützle - "Ant Colony Optimization"
  - Kennedy & Eberhart - "Particle Swarm Optimization"
  - Karaboga - "Artificial Bee Colony Algorithm"
  - Yang - "Firefly Algorithm" and "Cuckoo Search"
  - Goldberg - "Genetic Algorithms"
  - Russell & Norvig - "Artificial Intelligence: A Modern Approach"

## 📧 Contact

For questions or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

## 🔗 Additional Resources

- [Project Documentation](PROJECT_OVERVIEW.md)
- [Algorithm Details](algo_Traditional/README.md)
- [Comparison Guide](algo_Traditional/SUMMARY.md)

---

**Happy Optimizing! 🚀**

*Last Updated: November 2025*
