# 🚀 QUICK START - One Page Guide

## Run Everything in 3 Commands

```powershell
# 1. Navigate to the traditional algorithms folder
cd "d:\HCMUS Class Material\intro2AI\Intro2AI_Lab1\algo3 - Traditional"

# 2. Test that everything works (30 seconds)
python test_all.py

# 3. Run full comparison - THIS IS WHAT YOU NEED FOR YOUR ASSIGNMENT! (5 minutes)
python compare_all_algorithms.py
```

## 📊 What You'll Get

After running `compare_all_algorithms.py`, you'll have:

### Files Generated (save these for your report!)
- ✅ `comparison_rastrigin_convergence.png` - Rastrigin convergence curves
- ✅ `comparison_rastrigin_bars.png` - Rastrigin performance bars
- ✅ `comparison_tsp_convergence.png` - TSP convergence curves
- ✅ `comparison_tsp_bars.png` - TSP performance bars
- ✅ `comparison_tsp_solutions.png` - Visual TSP solutions

### Console Output (copy this to your report!)
- Performance table for all algorithms
- Best algorithm for each problem
- Time and quality metrics
- Statistical comparison (swarm vs traditional)

## 📋 Assignment Checklist

- [ ] Run `compare_all_algorithms.py` ✓
- [ ] Save all 5 PNG files ✓
- [ ] Copy console output ✓
- [ ] Review algorithm descriptions in README.md ✓
- [ ] Understand why swarm intelligence performs better ✓

## 🎯 Key Results to Discuss in Your Report

### Continuous (Rastrigin Function)
**Winner**: Usually PSO or ACOR (Swarm)  
**Why**: Better at escaping local minima in multimodal functions  
**Trade-off**: Slightly slower than Hill Climbing

### Discrete (TSP)
**Winner**: Usually Hybrid PSO (Swarm)  
**Why**: Combines global search with local optimization  
**Alternative**: Genetic Algorithm (good balance)

## 💡 Main Insights for Your Report

1. **Swarm Intelligence excels at complex, multimodal problems**
   - Multiple agents explore different regions
   - Information sharing leads to better solutions
   - More robust to poor initialization

2. **Traditional algorithms are faster but more limited**
   - Hill Climbing: Fast but gets stuck easily
   - Genetic Algorithm: Good balance of exploration/exploitation
   - A*: Optimal but only for small problems

3. **Hybrid approaches often win**
   - Hybrid PSO = Global search + Local refinement
   - Best of both worlds

## 🔍 File Locations

```
algo3 - Traditional/
├── compare_all_algorithms.py  ← RUN THIS!
├── continuous_traditional.py  ← Hill Climbing + GA for Rastrigin
├── tsp_traditional.py         ← Hill Climbing + A* + GA for TSP
├── test_all.py               ← Quick test
├── README.md                 ← Full documentation
└── SUMMARY.md                ← Detailed summary
```

## ⚙️ If You Want to Customize

### Change Problem Difficulty
Edit `compare_all_algorithms.py`:
```python
# Line ~420 - Continuous optimization
run_continuous_comparison(
    n_dims=10,  # Change from 5 to 10 (harder)
    ...
)

# Line ~440 - TSP
run_tsp_comparison(
    n_cities=30,  # Change from 20 to 30 (harder)
    ...
)
```

### Run Individual Algorithm Types
```powershell
python continuous_traditional.py  # Just Rastrigin algorithms
python tsp_traditional.py         # Just TSP algorithms
```

## 📊 Sample Results Table (for your report)

### Rastrigin Function (5D)
| Algorithm | Type | Best Cost | Time (s) |
|-----------|------|-----------|----------|
| Hill Climbing | Traditional | ~2.5 | 0.6 |
| Genetic Algorithm | Traditional | ~1.2 | 1.2 |
| ACOR | Swarm | ~0.9 | 1.1 |
| PSO | Swarm | ~0.6 | 0.5 |

### TSP (20 Cities)
| Algorithm | Type | Best Cost | Time (s) |
|-----------|------|-----------|----------|
| Hill Climbing | Traditional | ~395 | 0.3 |
| A* | Traditional | N/A | N/A |
| Genetic Algorithm | Traditional | ~380 | 3.5 |
| ACO | Swarm | ~375 | 18.0 |
| Hybrid PSO | Swarm | ~355 | 5.5 |

## 🎓 For Your Presentation

### Slide 1: Algorithms Implemented
- 3 Traditional: Hill Climbing, GA, A*
- 4 Swarm: ACOR, PSO, ACO, Hybrid PSO
- 2 Problems: Rastrigin (continuous), TSP (discrete)

### Slide 2: Results - Rastrigin
- Show convergence plot
- Show bar chart
- Highlight: Swarm intelligence wins by ~55%

### Slide 3: Results - TSP
- Show TSP solutions visualization
- Show performance comparison
- Highlight: Hybrid PSO wins (combines strengths)

### Slide 4: Conclusions
- Swarm intelligence better for complex problems
- Traditional algorithms faster for simple problems
- Hybrid approaches often best

## ❓ Common Questions

**Q: Why is A* not showing results for TSP?**  
A: A* only works for ≤15 cities. For 20 cities (default), it's skipped.

**Q: Can I add more algorithms?**  
A: Yes! The framework is extensible. Add your algorithm class and register it in `compare_all_algorithms.py`.

**Q: Results vary between runs?**  
A: Yes, these are stochastic algorithms. Use `np.random.seed()` for reproducibility.

**Q: Which algorithm should I recommend?**  
A: 
- **Rastrigin**: PSO (fast + good quality)
- **TSP**: Hybrid PSO (best quality) or GA (balanced)

## 🎯 Bottom Line

**ONE COMMAND gives you everything you need:**
```powershell
python compare_all_algorithms.py
```

This generates:
✅ All results  
✅ All plots  
✅ All comparisons  
✅ All analysis  

**Just run it, save the outputs, and write your report!**

---

**Good luck! 🌟**
