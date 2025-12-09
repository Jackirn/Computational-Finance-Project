# Computational Finance Project – Exam Version C

This repository contains the full implementation of the **Computational Finance** project for the course examination (Version C).  
The project develops and compares multiple portfolio optimization methodologies — classical, robust, factor-based, diversification-based, and deep-learning driven — using a universe of **16 synthetic equity indices**.

All results are generated through the main MATLAB script: Comp_Fin_Group9

---

## 📌 Project Overview

The project is structured into **five main exercises**, each addressing a different quantitative finance technique:

1. **Constrained Mean–Variance Frontier** and **Robust Frontier** via resampling  
2. **Black–Litterman Model** with equilibrium returns and investor views  
3. **Diversification-Based Portfolios** (Maximum Diversification Ratio & Maximum Entropy)  
4. **PCA-Based Risk Modelling** and **CVaR Optimization**  
5. **Personal Deep-Learning Allocation Strategy**

All models are estimated in-sample (2018–2022) and evaluated with out-of-sample analysis (2023–2024 where applicable).

---

## 🔧 Requirements

This project requires **MATLAB R2021a or later**, with:

- Optimization Toolbox  
- Statistics & Machine Learning Toolbox  
- Deep Learning Toolbox (for Exercise 5)

No Python dependencies are used — the project is entirely MATLAB-based.

---

## ▶️ How to Run

1. Open MATLAB  
2. Set the project root folder as the current directory  
3. Run:

```matlab
Comp_Fin_Group9
```

The script will automatically:
	•	load data
	•	run Exercises 1–5
	•	print portfolio summaries
	•	generate all figures in separate windows

⸻

## Exercise Summary

### Exercise 1 — Mean–Variance & Robust Frontier
	•	Constrained efficient frontier (A–B)
	•	Robust frontier using 200 resampled scenarios (C–D)

### Exercise 2 — Black–Litterman Model
	•	Reverse optimization to compute equilibrium returns
	•	3 investor views
	•	Posterior returns and updated frontier (E–F)

### Exercise 3 — Diversification-Based Portfolios
	•	Maximum Diversification Ratio (G)
	•	Maximum Entropy in Risk Contributions (H)
	•	Sector exposure constraints enforced

### Exercise 4 — PCA & CVaR Optimization
	•	PCA reconstruction of the covariance matrix
	•	Portfolio I: PCA-based maximum Sharpe (with volatility cap)
	•	Portfolio J: Minimum CVaR with target volatility 10% (shown to be infeasible → solved at the volatility boundary)

### Exercise 5 — Personal Strategy

A deep-learning allocator based on:
	•	600-day rolling window of returns
	•	Feed-forward neural network
	•	Softmax-constrained weights
	•	Objective: maximize risk-adjusted returns with regularization
	•	Tested out-of-sample on 2023–2024

⸻

## Output

Running the project generates:
	•	Efficient frontiers (standard + robust)
	•	Black–Litterman posterior returns and portfolios
	•	MDR and Entropy frontiers
	•	PCA explained variance plots
	•	CVaR-optimized portfolio
	•	Formatted tables summarizing:
	•	expected return
	•	volatility
	•	Sharpe ratio
	•	concentration metrics
	•	top asset exposures

All figures can optionally be saved inside the Plots/ directory.

⸻

## Authors

Giacomo Kirn, Riccardo Girgenti, Filippo Cuoghi, Francesco Ligorio, Luigi di Gregorio
Computational Finance – Exam Version C
MSc Mathematical Engineering
Politecnico di Milano — 2025