"""
Assignment 2 - discrete-event simulator for one airport security lane.

Event-driven (from scratch, heapq future-event list), FCFS, c identical servers,
Poisson arrivals, truncated-Normal service times (negative draws are resampled).

    simulate(lam, mu_b, sigma_b, c, n_pass, rng) -> waiting times in queue (min), in service order
"""
import heapq
import numpy as np

SEED = 20261113
ARRIVAL, DEPARTURE = 0, 1


def trunc_normal(rng, mean, sd, size):
    """Normal(mean, sd) conditioned on >= 0, by resampling negative draws."""
    x = rng.normal(mean, sd, size)
    bad = x < 0
    while bad.any():
        x[bad] = rng.normal(mean, sd, bad.sum())
        bad = x < 0
    return x


def simulate(lam, mu_b, sigma_b, c, n_pass, rng):
    """Simulate until n_pass passengers have STARTED and finished service.

    Returns an array of length n_pass: Wq of each passenger, indexed by arrival
    order (FCFS, so arrival order = service-start order).
    """
    inter = rng.exponential(1.0 / lam, n_pass)
    service = trunc_normal(rng, mu_b, sigma_b, n_pass)
    arr_times = np.cumsum(inter)

    fel = []                       # future event list: (time, type, id)
    heapq.heappush(fel, (arr_times[0], ARRIVAL, 0))
    queue = []                     # FIFO of waiting passenger ids (list + head pointer)
    head = 0
    busy = 0
    wait = np.empty(n_pass)
    served = 0

    while served < n_pass:
        t, kind, i = heapq.heappop(fel)
        if kind == ARRIVAL:
            if i + 1 < n_pass:
                heapq.heappush(fel, (arr_times[i + 1], ARRIVAL, i + 1))
            if busy < c:                     # free server: start immediately
                busy += 1
                wait[i] = 0.0
                heapq.heappush(fel, (t + service[i], DEPARTURE, i))
            else:
                queue.append(i)
        else:                                # departure
            served += 1
            if head < len(queue):            # next in line starts service
                j = queue[head]; head += 1
                wait[j] = t - arr_times[j]
                heapq.heappush(fel, (t + service[j], DEPARTURE, j))
            else:
                busy -= 1
    return wait


def simulate_lindley(lam, mu_b, sigma_b, c, n_pass, rng):
    """Independent check: Kiefer-Wolfowitz workload recursion (Lindley for c=1).
    Uses the same random numbers in the same order as simulate()."""
    inter = rng.exponential(1.0 / lam, n_pass)
    service = trunc_normal(rng, mu_b, sigma_b, n_pass)
    arr = np.cumsum(inter)
    free = np.zeros(c)             # time each server becomes free
    wait = np.empty(n_pass)
    for i in range(n_pass):
        k = np.argmin(free)
        start = max(arr[i], free[k])
        wait[i] = start - arr[i]
        free[k] = start + service[i]
    return wait


def rep_rngs(n, seed=SEED):
    """n independent streams (SeedSequence.spawn) - one per replication."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]
