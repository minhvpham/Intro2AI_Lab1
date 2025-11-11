# Algorithm Comparison Metrics - Update Summary

## Overview
The `compare_all_algorithms.py` script has been updated to include comprehensive new metrics for algorithm comparison as requested:

1. **Convergence Speed**
2. **Computational Complexity (Time and Space)**
3. **Robustness (Performance across multiple runs)**
4. **Scalability (Performance with problem size)**

## What's New

### 1. Enhanced AlgorithmResults Class

The `AlgorithmResults` class now includes:

- **Multiple Runs Support**: Store results from multiple independent trials
  - `add_run()`: Add results from each trial
  - `compute_robustness_metrics()`: Calculate statistics across runs
  
- **Convergence Speed Metric**: 
  - `compute_convergence_speed()`: Calculates iterations needed to reach 90% of final solution quality
  - Lower values indicate faster convergence

- **Memory Usage Tracking**: Peak memory usage during execution (requires psutil package)

### 2. New Analysis Functions

#### Comprehensive Metrics Table
```python
print_comprehensive_metrics_table(results, problem_name)
```
Displays all metrics in a single comprehensive table including:
- Best Cost
- Computation Time
- Convergence Speed (iterations to 90% solution)
- Robustness (Coefficient of Variation)
- Memory Usage

#### Computational Complexity Analysis
```python
print_complexity_analysis(results, problem_name)
```
Shows theoretical time and space complexity for each algorithm with notation explanation.

#### Robustness Analysis Visualization
```python
plot_robustness_analysis(results, problem_name, save_path)
```
Creates box plots showing:
- Solution quality distribution across multiple runs
- Computation time distribution
- Visual comparison of algorithm consistency

#### Convergence Speed Comparison
```python
plot_convergence_speed_comparison(results, problem_name, save_path)
```
Bar chart comparing how many iterations each algorithm needs to reach near-optimal solutions.

#### Scalability Analysis
```python
run_scalability_analysis(algorithm_runner, problem_sizes, problem_name, save_path)
```
Tests algorithms across different problem sizes and visualizes:
- Cost scaling with problem size
- Time scaling with problem size
- Identifies algorithms that maintain performance as problems grow

### 3. Multi-Trial Support

Both comparison functions now support running multiple independent trials:

```python
run_continuous_comparison(..., n_trials=5)
run_tsp_comparison(..., n_trials=5)
```

This enables robust statistical analysis:
- Mean and standard deviation of costs
- Coefficient of Variation (CV = std/mean)
- Success rate
- Best/worst case performance

### 4. Updated Main Function

The `main()` function now:

1. Runs algorithms with **5 independent trials** (configurable via `N_TRIALS`)
2. Generates comprehensive metrics tables
3. Shows complexity analysis
4. Creates additional visualizations:
   - Convergence speed comparison
   - Robustness analysis (box plots)
   - Scalability analysis (multiple problem sizes)

### 5. New Visualization Outputs

The script now generates **10 plots** instead of 5:

**Continuous Optimization (Rastrigin):**
- `comparison_rastrigin_convergence.png` - Convergence curves
- `comparison_rastrigin_bars.png` - Cost and time comparison
- `comparison_rastrigin_convergence_speed.png` - **NEW** Iterations to solution
- `comparison_rastrigin_robustness.png` - **NEW** Box plots of multiple runs
- `comparison_rastrigin_scalability.png` - **NEW** Performance vs dimension size

**Discrete Optimization (TSP):**
- `comparison_tsp_convergence.png` - Convergence curves
- `comparison_tsp_bars.png` - Cost and time comparison  
- `comparison_tsp_convergence_speed.png` - **NEW** Iterations to solution
- `comparison_tsp_robustness.png` - **NEW** Box plots of multiple runs
- `comparison_tsp_scalability.png` - **NEW** Performance vs number of cities
- `comparison_tsp_solutions.png` - Visual tour comparison

## Metrics Definitions

### 1. Convergence Speed
**Definition**: Number of iterations required to reach 90% of the final solution quality.

**Interpretation**: 
- Lower is better (faster convergence)
- Indicates how quickly an algorithm approaches optimal solutions
- Important for time-critical applications

### 2. Computational Complexity

**Time Complexity**: 
- Theoretical worst-case runtime as a function of problem parameters
- Notation: O(n), where n can be iterations (i), population (p), dimensions (d), etc.

**Space Complexity**:
- Memory requirements relative to problem size
- Important for large-scale problems

**Common Patterns**:
- Traditional: O(n × k × d) where k is restarts/evaluations
- Swarm: O(i × p × d) or O(i × p × n²) for TSP

### 3. Robustness

**Coefficient of Variation (CV)**: `std_dev / mean`

**Interpretation**:
- Lower CV = more consistent/robust
- CV < 0.1: Highly robust
- 0.1 < CV < 0.3: Moderately robust
- CV > 0.3: High variability

**Why It Matters**:
- Some algorithms are stochastic and vary across runs
- Robustness indicates reliability in practice
- Important for production systems requiring consistent performance

### 4. Scalability

**Definition**: How algorithm performance changes with problem size

**Metrics Tracked**:
- Solution quality vs problem size
- Computation time vs problem size
- Both plotted on same figure for easy comparison

**Interpretation**:
- Good scalability: Gradual increase in time, stable quality
- Poor scalability: Exponential time increase or quality degradation
- Essential for understanding algorithm limits

## Usage Example

```python
# Run comparison with new metrics
python compare_all_algorithms.py
```

The script will:
1. Run 5 trials for each algorithm (configurable)
2. Print comprehensive metrics tables
3. Show complexity analysis
4. Generate all 10 visualization plots
5. Provide detailed summary with insights

## Configuration

You can adjust the analysis parameters in `main()`:

```python
# Number of independent trials for robustness
N_TRIALS = 5  # Increase for more robust statistics

# Scalability test sizes
problem_sizes=[2, 5, 10, 15]  # For continuous
problem_sizes=[10, 20, 30, 40]  # For TSP
```

## Key Findings Format

The final summary now includes:

**For each problem type:**
- ✓ Best Solution Quality (algorithm name, cost, time)
- ✓ Fastest Convergence (algorithm name, iterations)
- ✓ Most Robust (algorithm name, CV score)
- ✓ Category Comparison (Swarm vs Traditional)
  - Average solution quality
  - Average computation time
  - Average convergence speed
  - Percentage improvement

## Benefits of New Metrics

1. **Convergence Speed**: Identifies algorithms that find good solutions quickly
2. **Complexity Analysis**: Predicts performance on larger problems
3. **Robustness**: Ensures reliability in production environments
4. **Scalability**: Guides algorithm selection based on expected problem sizes

## Requirements

### Required Packages (already in use):
- numpy
- matplotlib
- All existing algorithm implementations

### Optional Package:
- `psutil` - For memory usage tracking (install via: `pip install psutil`)
  - If not installed, memory metrics will show "N/A"

## Notes

- Multi-trial support is fully implemented for Hill Climbing and Genetic Algorithm
- Other algorithms run with n_trials parameter but can be extended similarly
- All algorithms now compute convergence speed
- Scalability analysis runs single trial per size to save time
- Results are automatically saved as high-resolution PNG files

## Theoretical Complexity Reference

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Hill Climbing | O(n × k × d) | O(d) | n=iterations, k=restarts |
| Genetic Algorithm | O(g × p × d) | O(p × d) | g=generations, p=population |
| ACOR | O(i × k × d²) | O(k × d) | k=archive size |
| PSO | O(i × p × d) | O(p × d) | p=particles |
| ACO (TSP) | O(i × a × n²) | O(a × n) | a=ants, n=cities |
| Firefly | O(i × n² × d) | O(n × d) | n=fireflies |
| ABC | O(i × n × d) | O(n × d) | n=food sources |
| Cuckoo Search | O(i × n × d) | O(n × d) | n=nests |

Where: i = iterations, d = dimensions, n = problem size

---

**Last Updated**: November 10, 2025
**Version**: 2.0
