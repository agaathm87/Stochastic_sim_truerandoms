"""
Assignment 2 - checks that the simulator implements the equations correctly.

Prints PASS/FAIL to the console only (no figures, no tables). It does NOT run the
assignment scenarios: P-K is checked at rho = 0.5 and 0.7 and for M/M/1 and M/D/1,
not at the Part 2A test point rho = 0.85.

Run:  python check_simulator.py      (a few seconds)
"""
import numpy as np
import pandas as pd
from scipy import stats

import sim

R, MU_B = 40, 1.0
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58} {detail}")


def mean_ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def in_ci(name, theory, x):
    m, lo, hi = mean_ci(x)
    report(name, lo <= theory <= hi, f"theory {theory:.4f}  sim {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


def lindley(A, B):
    """Vectorised Lindley recursion: W_n = S_n - min(0, min_k S_k)."""
    S = np.concatenate(([0.0], np.cumsum(B[:-1] - A[1:])))
    return S - np.minimum.accumulate(np.minimum(S, 0.0))


# 1. event-driven simulator == Kiefer-Wolfowitz / Lindley recursion on the same random numbers
for c, lam, n in [(1, 0.6, 50_000), (1, 3.0, 3000), (2, 1.2, 50_000), (2, 3.0, 3000), (3, 2.0, 20_000)]:
    we = sim.simulate(lam, MU_B, 0.25, c, n, np.random.default_rng(c * 10 + n))
    wl = sim.simulate_lindley(lam, MU_B, 0.25, c, n, np.random.default_rng(c * 10 + n))
    d = np.abs(we - wl).max()
    report(f"event sim == recursion (c={c}, rho={lam / c:.2f})", d < 1e-9, f"max |diff| {d:.1e}")

# 2. inputs: exponential inter-arrivals, truncated-normal service
rng = np.random.default_rng(1)
lam = 2.0
in_ci("mean inter-arrival = 1/lambda", 1 / lam,
      [rng.exponential(1 / lam, 50_000).mean() for _ in range(R)])
p = stats.kstest(rng.exponential(1 / lam, 100_000), "expon", args=(0, 1 / lam)).pvalue
report("inter-arrivals ~ Exp(lambda) (KS)", p > 0.05, f"p = {p:.3f}")
for sd in (0.25, 0.10):
    B = [sim.trunc_normal(rng, MU_B, sd, 100_000) for _ in range(R)]
    in_ci(f"E[B] = mu_B (sigma={sd})", MU_B, [b.mean() for b in B])
    in_ci(f"E[B^2] = mu_B^2 + sigma^2 (sigma={sd})", MU_B**2 + sd**2, [(b**2).mean() for b in B])
    report(f"no negative service times (sigma={sd})", min(b.min() for b in B) >= 0, "")
sd = 0.5                                              # larger sigma so redraws are frequent enough to test
x = rng.normal(MU_B, sd, 2_000_000)
in_ci("P(redraw) = Phi(-mu_B/sigma) (sigma=0.5)", stats.norm.cdf(-MU_B / sd),
      (x < 0).reshape(R, -1).mean(1))

# 3. P-K at points that are NOT the assignment's test point
def pk(rho, cb2):
    return rho / (1 - rho) * (1 + cb2) / 2 * MU_B


for rho, kind in [(0.5, "normal"), (0.7, "normal"), (0.7, "exp"), (0.7, "det")]:
    ms = []
    for g in sim.rep_rngs(R, 4242 + int(rho * 100) + len(kind)):
        A = g.exponential(1 / rho, 101_000)
        B = {"normal": lambda: sim.trunc_normal(g, MU_B, 0.25, 101_000),
             "exp": lambda: g.exponential(MU_B, 101_000),
             "det": lambda: np.full(101_000, MU_B)}[kind]()
        ms.append(lindley(A, B)[1000:].mean())
    cb2 = {"normal": 0.25**2, "exp": 1.0, "det": 0.0}[kind]
    label = {"normal": "M/G/1", "exp": "M/M/1", "det": "M/D/1"}[kind]
    in_ci(f"P-K {label}, rho={rho}", pk(rho, cb2), ms)

# 3b. Little's law: queue length seen by arriving passengers (PASTA) = lambda * Wq
lq, wq = [], []
for g in sim.rep_rngs(R, 5151):
    lam_l = 0.7
    A = g.exponential(1 / lam_l, 101_000); B = sim.trunc_normal(g, MU_B, 0.25, 101_000)
    W = lindley(A, B); T = np.cumsum(A); S = T + W              # FCFS: start times are sorted
    idx = np.arange(len(T))
    waiting = idx - np.minimum(np.searchsorted(S, T, side="right"), idx)
    lq.append(waiting[1000:].mean()); wq.append(lam_l * W[1000:].mean())
in_ci("Little: L_q seen by arrivals = lambda * Wq (rho=0.7)", np.mean(wq), lq)

# 4. overload: when rho > 1 the waits grow by about E[B]/c - 1/lambda per passenger
for c, lam in [(1, 2.0), (2, 3.0)]:
    W = np.array([sim.simulate(lam, MU_B, 0.25, c, 3000, g) for g in sim.rep_rngs(R, 77 + c)])
    slope = np.polyfit(np.arange(1, 3001), W.mean(0), 1)[0]
    th = MU_B / c - 1 / lam
    report(f"fluid slope E[B]/c - 1/lambda (c={c}, rho={lam / c:.1f})", abs(slope - th) / th < 0.01,
           f"theory {th:.4f}  sim {slope:.4f}")

# 5. Welch test: two independent runs of the SAME overloaded system (lambda=2, c=1)
#    should reject H0 in about alpha = 5% of repeated experiments
rng = np.random.default_rng(99)
rej, n_rep = 0, 400
for _ in range(n_rep):
    m = []
    for _g in range(2 * R):
        A = rng.exponential(0.5, 3000); B = sim.trunc_normal(rng, MU_B, 0.25, 3000)
        m.append(lindley(A, B).mean())
    rej += stats.ttest_ind(m[:R], m[R:], equal_var=False).pvalue < 0.05
p = rej / n_rep; se = np.sqrt(0.05 * 0.95 / n_rep)
report("Welch A/A test: rejection rate = alpha", abs(p - 0.05) < 1.96 * se,
       f"{p:.3f} (expected 0.05 +/- {1.96 * se:.3f})")
x, y = rng.normal(0, 1, R), rng.normal(0, 3, R)
pooled = stats.ttest_ind(x, y, equal_var=True).statistic
welch = stats.ttest_ind(x, y, equal_var=False).statistic
report("Welch t == pooled t when both groups have R runs", abs(pooled - welch) < 1e-10, "")

print(f"\n{sum(results)} / {len(results)} checks pass "
      "(each 95% CI check can miss by chance about 1 time in 20)")
