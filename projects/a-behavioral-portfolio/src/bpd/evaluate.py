"""Step 6 of the six-step process: evaluating policies.

Three rules are imposed here rather than left to the caller.  The first is a recurring
error the course names, the second is required by the brief, and the third is ours.

Common random numbers.  Every policy in a comparison is run on the same array of
scenarios, since deck "005 SDA" page 18 states that reusing the paths is what makes the
differences something other than simulation noise, and page 24 lists "comparing on
different samples" among the recurring modelling errors.

Separate seeds.  The scenarios used to tune a policy's parameters and the scenarios used
to report its performance come from different seeds, so a parameter that happens to suit
one set of draws cannot flatter itself on the same draws.

Paired differences, which is ours.  Because the scenarios are shared, a policy and its
baseline are observed on the same draw, so the confidence interval that answers "is this
policy better" is the one around the mean of the per-scenario difference.  That interval
is narrower than the one around either mean whenever the two policies' outcomes are
positively correlated across scenarios, which sharing the draws makes likely without
guaranteeing it, and a test asserts the narrowing holds for the comparison reported.
"""
import numpy as np

from . import config

Z95 = 1.959963984540054


def scenarios(rets: np.ndarray, n_paths: int, seed: int, horizon: int = None):
    """One array of scenarios, drawn once and reused by every policy."""
    from . import data
    horizon = config.HORIZON_MONTHS if horizon is None else horizon
    return data.block_bootstrap(rets, n_paths, horizon, seed)


def summarise(sample: np.ndarray) -> dict:
    """Mean, spread and a confidence interval for the mean of one sample."""
    sample = np.asarray(sample, dtype=float)
    n = sample.size
    standard_error = sample.std(ddof=1) / np.sqrt(n)
    return {
        "n": n,
        "mean": float(sample.mean()),
        "std": float(sample.std(ddof=1)),
        "median": float(np.median(sample)),
        "p5": float(np.percentile(sample, 5)),
        "p95": float(np.percentile(sample, 95)),
        "se": float(standard_error),
        "ci_low": float(sample.mean() - Z95 * standard_error),
        "ci_high": float(sample.mean() + Z95 * standard_error),
    }


def paired_difference(sample: np.ndarray, baseline: np.ndarray) -> dict:
    """The per-scenario difference against a baseline, summarised.

    The two samples must come from the same scenarios in the same order, which is what
    common random numbers buys and what makes this interval meaningful.
    """
    sample = np.asarray(sample, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if sample.shape != baseline.shape:
        raise ValueError("a paired difference needs the same scenarios on both sides")
    difference = sample - baseline
    out = summarise(difference)
    out["wins"] = float((difference > 0).mean())
    out["significant"] = bool(out["ci_low"] > 0.0 or out["ci_high"] < 0.0)
    return out


def mandate_is_kept(out, max_position: float, quantile: float = 95.0) -> bool:
    """Whether the portfolio still obeys the Part 2 position cap at the end.

    Part 2 caps any single holding, but it imposes that cap once, at the moment the
    portfolio is built.  Nothing in the objective asks for it to hold afterwards, so a
    policy is free to let the weights drift past it, and the drift costs nothing in a
    sample where the asset that runs away is the one the optimiser already liked.

    Reading that cap as a standing constraint on the policy is OURS: the brief imposes
    it on the Part 2 portfolio and says nothing about the policies of Part 3.  The
    quantile is ours too, and 95% was chosen because a mandate that holds on average is
    not one, while requiring it in every scenario would admit only a rule that trades on
    the cap itself rather than on drift from the target.
    """
    largest = np.asarray(out["terminal_weights"]).max(axis=1)
    return bool(np.percentile(largest, quantile) <= max_position)


def tune(make_policy_for, grid, paths, target, client, reference, metric="value",
         feasible=None):
    """Grid search over a policy's parameters on the tuning scenarios.

    `grid` is a list of parameter tuples and `make_policy_for` turns one tuple into a
    policy factory, so the search is the course's own policy search: pick a structure,
    then choose theta inside it by simulation on shared paths.

    `feasible` is an optional predicate on the simulation output.  When it is given the
    best feasible setting is returned, and every setting is still reported with whether
    it passed, because the shape of the trade-off between the objective and the
    constraint is the finding rather than a detail of the search.
    """
    from . import simulator
    results = []
    for parameters in grid:
        out = simulator.run_policy(make_policy_for(parameters), paths, target,
                                   client, reference)
        largest = np.asarray(out["terminal_weights"]).max(axis=1)
        results.append({"parameters": parameters,
                        "score": float(np.mean(out[metric])),
                        "median_terminal": float(np.median(out["terminal"])),
                        "turnover_per_year": float(np.mean(out["turnover_per_year"])),
                        "largest_end_weight_p95": float(np.percentile(largest, 95)),
                        "feasible": True if feasible is None else bool(feasible(out))})
    admissible = [row for row in results if row["feasible"]]
    if not admissible:
        raise ValueError("no setting on this grid satisfies the constraint")
    best = max(admissible, key=lambda row: row["score"])
    return best, results
