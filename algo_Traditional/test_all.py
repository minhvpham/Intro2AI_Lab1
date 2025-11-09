"""
Quick Test Script - Verify All Implementations Work
"""

import numpy as np
import sys
import os

print("="*70)
print("TESTING ALL ALGORITHM IMPLEMENTATIONS")
print("="*70)

# Test continuous traditional algorithms
print("\n[1/2] Testing Continuous Traditional Algorithms...")
try:
    from continuous_traditional import HillClimbing, GeneticAlgorithm, rastrigin
    
    # Quick test with 2D
    hc = HillClimbing(rastrigin, 2, [-5.12, 5.12], 
                      step_size=0.5, max_iterations=50, n_restarts=3)
    hc_sol, hc_cost = hc.run()
    print(f"  ✓ Hill Climbing works! Best cost: {hc_cost:.4f}")
    
    ga = GeneticAlgorithm(rastrigin, 2, [-5.12, 5.12],
                          population_size=20, n_generations=30)
    ga_sol, ga_cost = ga.run()
    print(f"  ✓ Genetic Algorithm works! Best cost: {ga_cost:.4f}")
    
    print("  ✓ Continuous algorithms: PASSED")
except Exception as e:
    print(f"  ✗ Error in continuous algorithms: {e}")

# Test TSP traditional algorithms
print("\n[2/2] Testing TSP Traditional Algorithms...")
try:
    from tsp_traditional import (HillClimbingTSP, AStarTSP, GeneticAlgorithmTSP,
                                  create_cities)
    
    # Small test problem
    cities = create_cities(10, seed=42)
    
    hc_tsp = HillClimbingTSP(cities, n_restarts=3, max_iterations=100)
    hc_tour, hc_cost = hc_tsp.run()
    print(f"  ✓ Hill Climbing TSP works! Best cost: {hc_cost:.2f}")
    
    astar_tsp = AStarTSP(cities, time_limit=10)
    astar_tour, astar_cost = astar_tsp.run()
    print(f"  ✓ A* Search TSP works! Best cost: {astar_cost:.2f}")
    
    ga_tsp = GeneticAlgorithmTSP(cities, population_size=30, n_generations=50)
    ga_tour, ga_cost = ga_tsp.run()
    print(f"  ✓ Genetic Algorithm TSP works! Best cost: {ga_cost:.2f}")
    
    print("  ✓ TSP algorithms: PASSED")
except Exception as e:
    print(f"  ✗ Error in TSP algorithms: {e}")

print("\n" + "="*70)
print("ALL TESTS COMPLETED!")
print("="*70)
print("\nYou can now run:")
print("  - python continuous_traditional.py")
print("  - python tsp_traditional.py")
print("  - python compare_all_algorithms.py")
print("="*70)
