"""
Assignment 2 - run the experiments; returns raw numbers only.

No figures, tables or hypothesis tests are produced here: the results section
builds those from the arrays these functions return.

    from experiments import arrival_rate, part2a, part2b
    lam = arrival_rate("../airport.csv")         # pax/min per lane
    w2a = part2a()                               # (40,) replication means of Wq, after warm-up
    W2b = part2b(lam)                            # {"Baseline": (40, 3000), "Option A": ..., "Option B": ...}

Notation follows Equations_Assumptions.pdf.
"""
import numpy as np
import pandas as pd

import sim

# given parameters (assignment)
N_LANES, DAYS, HOURS = 50, 30, 16
MU_B = 1.0
R = 40
RHO_TARGET = 0.85
N0, N_KEEP = 1000, 100_000          # Part 2A: warm-up discarded, passengers kept per replication
N_2B = 3000                         # Part 2B: passengers per replication
SCENARIOS = {                       # name: (servers c, service sd sigma)
    "Baseline": (1, 0.25),
    "Option A": (2, 0.25),
    "Option B": (1, 0.10),
}


def arrival_rate(csv_path):
    """Part 1: lambda = mean September (2015-2019) total passengers / (30*16*50*60)."""
    air = pd.read_csv(csv_path)
    sep = air[(air.Month == "September") & air.Year.between(2015, 2019)]["Total Passengers"]
    return sep.mean() / (DAYS * HOURS * N_LANES * 60)


def part2a(sigma=0.25, seed=sim.SEED, return_waits=False):
    """Part 2A: M/G/1 at lambda_test = rho_target / E[B].
    Returns the R replication means of Wq over passengers n0+1 .. n0+n_keep.
    With return_waits=True, returns the individual waits instead: array (R, n_keep),
    warm-up already discarded."""
    lam_test = RHO_TARGET / MU_B
    out = []
    for g in sim.rep_rngs(R, seed):
        w = sim.simulate(lam_test, MU_B, sigma, 1, N0 + N_KEEP, g)[N0:]
        out.append(w if return_waits else w.mean())
    return np.array(out)


def part2b(lam, seed=sim.SEED):
    """Part 2B: transient runs from an empty lane, N_2B passengers, R replications per scenario.
    Independent random streams per scenario. Returns {scenario: array (R, N_2B) of Wq}."""
    res = {}
    for j, (name, (c, sd)) in enumerate(SCENARIOS.items()):
        res[name] = np.array([sim.simulate(lam, MU_B, sd, c, N_2B, g)
                              for g in sim.rep_rngs(R, seed + 1000 * (j + 1))])
    return res


def part2b_common(lam, seed=sim.SEED + 5000):
    """Same as part2b but every scenario reuses the same R seeds (common random numbers),
    for an optional paired comparison."""
    return {name: np.array([sim.simulate(lam, MU_B, sd, c, N_2B, g)
                            for g in sim.rep_rngs(R, seed)])
            for name, (c, sd) in SCENARIOS.items()}


if __name__ == "__main__":
    import os
    lam = arrival_rate(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "airport.csv"))
    w2a = part2a()
    W2b = part2b(lam)
    print("simulation finished:")
    print(f"  Part 2A: {w2a.shape[0]} replication means")
    for k, v in W2b.items():
        print(f"  Part 2B {k}: waits array {v.shape}")
