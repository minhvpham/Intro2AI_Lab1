# PSO Parameters: Theory and Practical Guide

## 🎯 Core Concepts

### The PSO Velocity Update Equation
```
v(t+1) = w·v(t) + c1·r1·(pBest - x(t)) + c2·r2·(gBest - x(t))
         ↑         ↑                       ↑
      Inertia   Cognitive              Social
```

This equation controls how particles move through the search space.

---

## 📊 Parameter Deep Dive

### 1. Inertia Weight (w) - The Momentum Controller

**Mathematical Role**: Scales the previous velocity
```
Inertia Component = w · v(t)
```

**Physical Interpretation**: 
- Particle "mass" or "momentum"
- How much past motion influences current motion

#### High w (0.7 - 0.9): Exploration 🔭
```
✓ Strong momentum
✓ Large velocity updates
✓ Can "fly over" local minima
✓ Explores distant regions
✗ May overshoot and fail to converge
✗ Can "fly out of bounds"
```

**Use when**:
- Problem has many local minima (multimodal)
- Search space is large
- Global exploration is priority
- Example: Rastrigin, Ackley, Griewank

#### Low w (0.4 - 0.6): Exploitation 🎯
```
✓ Quick "braking"
✓ Velocity diminishes rapidly
✓ Focuses on current region
✓ Fast convergence
✗ Easily trapped in local minima
✗ Stops moving too quickly
```

**Use when**:
- Problem has single optimum (unimodal)
- Need fast convergence
- Refinement/fine-tuning phase
- Example: Sphere, Quadratic functions

#### Mathematical Impact
```python
# High w example (w=0.9)
if v(t) = 10:
    inertia = 0.9 * 10 = 9  # Strong carryover

# Low w example (w=0.4)  
if v(t) = 10:
    inertia = 0.4 * 10 = 4  # Rapid dampening
```

---

### 2. Cognitive Coefficient (c1) - The Individual Explorer

**Mathematical Role**: Controls pull toward personal best
```
Cognitive Force = c1 · r1 · (pBest - x)
```

**Psychological Interpretation**:
- Particle's "trust in own experience"
- "Individualistic" vs "conformist"
- Personal memory influence

#### High c1 (1.5 - 2.5): Strong Individualism 🎨
```
✓ Each particle searches independently
✓ High swarm diversity
✓ Reduces "herd mentality"
✓ Better exploration
✗ Weak collective intelligence
✗ May not converge if too high
```

**Effect on swarm**:
- Particles spread out
- Each follows own promising regions
- Swarm looks "scattered"
- Less prone to premature convergence

#### Low c1 (0.5 - 1.0): Weak Individualism 🐑
```
✓ Less individual wandering
✓ More influenced by gBest
✓ Tighter swarm formation
✗ Reduced diversity
✗ Risk of collective error
```

**Effect on swarm**:
- Particles cluster together
- Follow collective wisdom
- Swarm looks "cohesive"

#### Balance with c2
```python
# High exploration: c1 > c2
c1 = 2.0  # Trust own experience
c2 = 1.0  # Less trust in swarm
# → Diverse, independent search

# High exploitation: c1 < c2  
c1 = 1.0  # Less own exploration
c2 = 2.5  # Strong trust in swarm
# → Fast convergence to consensus
```

---

### 3. Social Coefficient (c2) - The Swarm Consensus

**Mathematical Role**: Controls pull toward global best
```
Social Force = c2 · r2 · (gBest - x)
```

**Psychological Interpretation**:
- Swarm's "collective intelligence"
- "Herd instinct" strength
- Trust in group consensus

#### High c2 (2.0 - 3.0): Strong Conformity 🐝
```
✓ Fast convergence
✓ Strong collective pull
✓ Efficient for simple problems
✗ "Herd collapse" to gBest
✗ MAJOR risk: premature convergence
✗ All particles cluster at one point
```

**Effect on swarm**:
- Rapid movement toward gBest
- All particles converge quickly
- Swarm looks like "collapsing star"
- ⚠️ **DANGER**: May lock onto first local minimum!

#### Low c2 (0.5 - 1.5): Weak Conformity 🦅
```
✓ Maintains diversity
✓ Avoids premature convergence
✓ Better for complex problems
✗ Slower convergence
✗ May not exploit good solutions
```

**Effect on swarm**:
- Gradual movement toward gBest
- Particles maintain some independence
- Swarm looks "spread out"

#### The Premature Convergence Problem
```
c2 >> c1 → DANGER!

Example: c1=0.5, c2=3.0
1. One particle finds mediocre solution
2. Becomes gBest (only one found so far)
3. All particles RUSH toward it (high c2)
4. No diversity left to explore
5. TRAPPED in local minimum!

This is the #1 cause of PSO failure on complex problems!
```

---

## ⚖️ The Balance: Exploration vs Exploitation

### The Fundamental Trade-off

```
EXPLORATION          BALANCED           EXPLOITATION
    🔭                  ⚖️                   🎯
    
Search widely       Mix both          Focus locally
Find new regions    Adapt strategy    Refine solution
Avoid local min     Best overall      Fast converge
Slower              Medium speed      Faster
High diversity      Moderate          Low diversity

Parameters:         Parameters:        Parameters:
w = 0.7-0.9        w = 0.7            w = 0.4-0.6
c1 = 2.0-2.5       c1 = 1.5           c1 = 1.0-1.5
c2 = 1.0-1.5       c2 = 1.5           c2 = 2.0-2.5
c1 > c2            c1 ≈ c2            c1 < c2
```

### Mathematical Formulation

**Exploration Dominance** (c1 > c2, high w):
```
v ≈ large_w·v_old + large_c1·(pBest-x) + small_c2·(gBest-x)
    ↑                ↑                    ↑
    Big momentum    Big individual       Small social
    
→ Particle maintains velocity
→ Follows own discoveries
→ Less influenced by swarm
→ HIGH DIVERSITY
```

**Exploitation Dominance** (c2 > c1, low w):
```
v ≈ small_w·v_old + small_c1·(pBest-x) + large_c2·(gBest-x)
    ↑                ↑                    ↑
    Small momentum  Small individual     Big social
    
→ Particle slows down quickly
→ Ignores own discoveries
→ Strongly attracted to gBest
→ LOW DIVERSITY (convergence)
```

---

## 🎨 Parameter Interaction Effects

### Case Study 1: c2 >> c1 (Premature Convergence)
```python
w = 0.7, c1 = 0.5, c2 = 3.0

Iteration 1:
  particle_1 finds: fitness = 10 → becomes gBest
  
Iteration 2:
  ALL particles: social_force = 3.0 * (gBest - x)  # HUGE!
                 cognitive_force = 0.5 * (pBest - x)  # tiny
  → Everyone rushes to particle_1's position
  
Iteration 3:
  All particles clustered at same spot
  No diversity left to explore better regions
  STUCK at fitness = 10 (may not be global optimum!)
```

### Case Study 2: c1 >> c2 (Swarm Disintegration)
```python
w = 0.9, c1 = 3.0, c2 = 0.5

Effect:
  - Each particle strongly trusts own pBest
  - Weak trust in swarm's gBest
  - High momentum keeps them moving
  
Result:
  + High diversity maintained
  + Good exploration
  - Swarm "disintegrates" into N individual searchers
  - Loses "swarm intelligence" advantage
  - May fail to converge effectively
```

### Case Study 3: w too high (Divergence)
```python
w = 0.95, c1 = 1.5, c2 = 1.5

Effect:
  v(t+1) = 0.95·v(t) + ...
  
  If v(t) = 10:
    next: v = 0.95*10 + ... ≈ 9.5 + forces
    next: v ≈ 9.0 + forces
    ...
  
  Velocity barely decreases!
  Particles "fly" with huge velocities
  
Result:
  - Constantly overshoot optimum
  - "Bounce around" search space
  - Never converge (or very slowly)
```

### Case Study 4: w too low (Stagnation)
```python
w = 0.2, c1 = 1.5, c2 = 1.5

Effect:
  v(t+1) = 0.2·v(t) + ...
  
  If v(t) = 10:
    next: v = 0.2*10 + ... ≈ 2 + forces
    next: v ≈ 0.4 + forces
    ...
  
  Velocity rapidly approaches 0
  
Result:
  - Particles "brake" too quickly
  - Stop moving (v → 0)
  - TRAPPED in nearest local minimum
  - No exploration happened!
```

---

## 📐 Mathematical Analysis

### Convergence Condition (Clerc & Kennedy, 2002)

For PSO to converge, parameters must satisfy:
```
φ = c1 + c2 > 4

Constriction coefficient:
χ = 2 / |2 - φ - √(φ² - 4φ)|

Safe parameters:
w = χ
c1 = χ · c1'
c2 = χ · c2'

Classical choice: φ = 4.1, χ ≈ 0.729
→ w = 0.729, c1 = c2 = 1.49445
```

### Velocity Bounds
```
v_max = k · (x_max - x_min)

Common choices:
k = 0.1 to 0.2  (10-20% of search range)

Purpose:
- Prevent particles from "flying away"
- Control exploration step size
- Maintain search stability
```

---

## 🎓 Tuning Strategy

### Step-by-Step Parameter Tuning

#### Step 1: Identify Problem Type
```
Questions:
1. How many local minima? (Use test runs or literature)
2. Is landscape smooth or rugged?
3. What's the dimensionality?
4. Time constraints?

Decision:
- Simple/Unimodal → Favor EXPLOITATION
- Complex/Multimodal → Favor EXPLORATION
- Unknown → Start BALANCED
```

#### Step 2: Choose Initial Parameters
```python
# BALANCED (start here if unsure)
w = 0.729
c1 = 1.49445
c2 = 1.49445

# Or simplified:
w = 0.7
c1 = 1.5
c2 = 1.5
```

#### Step 3: Test and Diagnose
```python
# Run PSO and observe:

If premature convergence (stuck in local minimum):
    → INCREASE exploration:
      w → 0.8 or 0.9
      c1 → 2.0
      c2 → 1.0
      
If slow convergence (not finding good solution):
    → INCREASE exploitation:
      w → 0.5 or 0.4
      c2 → 2.0
      c1 → 1.0
      
If divergence (fitness getting worse):
    → DECREASE w:
      w → 0.6 or 0.5
      
If stagnation (particles stop moving):
    → INCREASE w:
      w → 0.7 or 0.8
```

#### Step 4: Fine-tune
```python
# Small adjustments:
w ± 0.1
c1 ± 0.2
c2 ± 0.2

# Test multiple times (PSO is stochastic!)
n_runs = 10  # Run 10 times, take average
```

---

## 📊 Problem-Specific Recommendations

### Unimodal Functions (Sphere, Quadratic)
```python
# Goal: Fast convergence
w = 0.4 - 0.6      # Low momentum
c1 = 1.0 - 1.5     # Moderate individual
c2 = 2.0 - 2.5     # High social
```

### Multimodal with Few Minima (Rosenbrock)
```python
# Goal: Balance speed and exploration
w = 0.6 - 0.7      # Moderate momentum
c1 = 1.5           # Balanced
c2 = 1.5           # Balanced
```

### Highly Multimodal (Rastrigin, Ackley, Griewank)
```python
# Goal: Avoid local minima
w = 0.7 - 0.9      # High momentum
c1 = 1.5 - 2.5     # High individual
c2 = 1.0 - 1.5     # Moderate social
```

### High-dimensional Problems (D > 30)
```python
# Goal: Maintain diversity in high-D space
w = 0.7 - 0.8      # Good momentum
c1 = 2.0           # Strong individual
c2 = 1.2           # Lower social
n_particles = 2*D  # More particles needed
```

---

## 🔬 Advanced Techniques

### 1. Adaptive Inertia Weight
```python
# Linear decrease
w(t) = w_max - (w_max - w_min) * t / T

# Common: w: 0.9 → 0.4
# Exploration early, exploitation late
```

### 2. Time-varying Acceleration Coefficients
```python
# Increase exploitation over time
c1(t) = c1_start - (c1_start - c1_end) * t / T
c2(t) = c2_start + (c2_end - c2_start) * t / T

# Example: c1: 2.5 → 0.5, c2: 0.5 → 2.5
```

### 3. Restart Mechanism
```python
# If diversity < threshold:
if diversity(swarm) < 0.01:
    reinitialize_positions()  # Restart search
    keep_gbest()  # Remember best found
```

---

## 📖 Summary Table

| Parameter | Range | Low Effect | High Effect | Simple Problem | Complex Problem |
|-----------|-------|------------|-------------|----------------|-----------------|
| **w** | 0.4-0.9 | Quick brake, focus | Strong momentum, explore | 0.4-0.6 | 0.7-0.9 |
| **c1** | 0.5-2.5 | Low diversity, conform | High diversity, individual | 1.0-1.5 | 1.5-2.5 |
| **c2** | 0.5-2.5 | Weak consensus, slow | Strong consensus, fast | 2.0-2.5 | 1.0-1.5 |
| **Balance** | - | - | - | c2 > c1 | c1 > c2 |

---

## ⚠️ Common Pitfalls

### 1. c2 Too High
```
SYMPTOM: Fast initial convergence, poor final solution
CAUSE: Premature convergence to first local minimum
FIX: Reduce c2, increase c1 and w
```

### 2. w Too High
```
SYMPTOM: Particles "bounce around", never settle
CAUSE: Velocity doesn't decrease
FIX: Reduce w to 0.6-0.7
```

### 3. w Too Low
```
SYMPTOM: Particles stop moving early, poor solution
CAUSE: Velocity goes to zero too quickly
FIX: Increase w to 0.6-0.7
```

### 4. Both c1 and c2 Too High
```
SYMPTOM: Unstable behavior, divergence
CAUSE: Forces too strong, violates convergence condition
FIX: Ensure c1 + c2 ≤ 4.0
```

---

## 🎯 Quick Reference

```
┌─────────────────────────────────────────────────────┐
│                  PSO PARAMETER CHEAT SHEET          │
├─────────────────────────────────────────────────────┤
│                                                      │
│  SAFE START:                                        │
│    w = 0.729, c1 = 1.494, c2 = 1.494               │
│                                                      │
│  NEED MORE EXPLORATION? (Complex problem)           │
│    ↑ w to 0.8-0.9                                   │
│    ↑ c1 to 2.0-2.5                                  │
│    ↓ c2 to 1.0-1.5                                  │
│                                                      │
│  NEED MORE EXPLOITATION? (Simple problem)           │
│    ↓ w to 0.4-0.6                                   │
│    ↓ c1 to 1.0-1.5                                  │
│    ↑ c2 to 2.0-2.5                                  │
│                                                      │
│  TROUBLESHOOTING:                                   │
│    Premature convergence → Increase exploration     │
│    Too slow → Increase exploitation                 │
│    Diverging → Decrease w                           │
│    Stagnant → Increase w                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

**Remember**: PSO is a **stochastic algorithm**. Always run multiple times and average results!
