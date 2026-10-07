# Derivative-limited radial-step prototype

Run `python scripts/benchmark_adaptive_steps.py` from the repository root.
This experiment creates temporary kernels; it does not alter the production solver.

Policy: reuse the existing four structure derivatives and limit the predicted fractional change in P, T, enclosed mass, and luminosity. Mass and luminosity scales have floors of 0.0001 Mstar and 0.001 Lstar. Cap steps at R/200; allow growth of at most 20% per shell; cap at half the remaining radius except for a final R/25000 floor. The surface initialization is unchanged for adaptive variants. No additional derivative evaluations or step retries are introduced. This is a change limiter, not an integration-error estimate.

Warm local benchmark, 50 solar-mass trials (a 7x7 neighborhood plus the exact default), X=.7, Z=.008. Compilation excluded. Single-model timings average 200 repetitions. Hardware and search region affect results.

| Policy | Default shells | Default time (ms) | 50-model time (ms) | Passing trials |
|---|---:|---:|---:|---:|
| Current | 981 | .639 | 36.70 | 1 |
| Finer fixed reference | 4901 | 3.491 | 185.92 | 0 |
| 2% change limit | 1596 | 1.063 | 59.88 | 1 |
| 5% change limit | 645 | .438 | 27.38 | 1 |
| 10% change limit | 348 | .231 | 18.02 | 2 |

The finer fixed reference uses R/5000 initially and in the main interior, R/25000 in the core, and 25000 shell capacity. It is a finer comparison, not established ground truth. The default L=.8652, Teff=5513.5 passes the current and adaptive solvers but fails the finer reference's density check.

Relative T/rho RMS differences on common sampled radii 0.03–0.95 R are .8195% for current, .8177% for 2%, .8209% for 5%, and .8324% for 10%. These aggregated interior-profile comparisons do not validate the extrapolated core. Status agreement with the finer reference is 43/50 for current, 2%, and 5%, and 42/50 for 10%.

The 5% prototype is the more conservative speed candidate: about 1.46x faster on the default and 1.34x on this grid. The 10% version is faster but changes an additional trial's passing status. Before adopting any policy, test multiple masses and compositions and examine convergence of core diagnostics with finer steps, including sensitivity to surface initialization. Keep the current app solver until that validation is complete.
