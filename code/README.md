# Assignment 2: simulation code

Python 3 with numpy, scipy and pandas. Run from inside `code/`; `airport.csv` sits one folder up. This code makes **no figures, tables or hypothesis tests**: it only simulates and returns raw numbers. Notation follows `Equations_Assumptions.pdf` and the brief. In the code, `mu_b` = E[B], `sigma_b` = σ, `lam` = λ, `c` = servers, `N_LANES` = N = 50, `N_2B` = K = 3000 and `R` = 40.

| File | What it does |
|---|---|
| `sim.py` | Discrete-event simulator for one lane: Poisson arrivals, truncated-normal service (negative draws redrawn), FCFS, `c` servers. `simulate(lam, mu_b, sigma_b, c, n_pass, rng)` returns each passenger's wait in queue, in arrival order. `simulate_lindley(...)` is the same model written as the Kiefer–Wolfowitz/Lindley recursion. `rep_rngs(n, seed)` gives `n` independent random streams, one per replication. |
| `experiments.py` | `arrival_rate(csv)` gives Part 1 λ. `part2a()` returns the 40 replication means of W_q after discarding 1000 warm-up passengers, at λ_test = 0.85; `part2a(return_waits=True)` returns the individual waits (40 × 100 000). `part2b(lam)` returns `{scenario: (40, 3000) array of waits}` for Baseline, Option A and Option B. `part2b_common(lam)` does the same with common random numbers, for an optional paired test. |
| `check_simulator.py` | Checks that the code implements the equations (23 checks, PASS/FAIL printed to the console). It confirms that the event simulator matches the recursion exactly, that the arrival and service distributions are right, that P-K holds at ρ = 0.5 and 0.7 and for M/M/1 and M/D/1, and that the overload slope, Little's law and the Welch test behave as they should. It deliberately does not run the assignment's own scenarios. |

Example:

```python
from experiments import arrival_rate, part2a, part2b
lam = arrival_rate("../airport.csv")
w2a = part2a()           # compare with the P-K value, t-test, CI ...
W2b = part2b(lam)        # W2b["Option A"].mean(axis=1) -> 40 replication means
```

Seeds are fixed (`sim.SEED`), so every run returns identical numbers.
