"""Command-line driver for Project B: tune, evaluate, and write every table.

    python run_analysis.py [--eval 500] [--tune 200] [--quick]

Writes CSVs to results/ and a plain-text digest to results/summary.txt.
"""
import argparse
import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from dtp import config, policies, simulator  # noqa: E402

RESULTS = HERE / "results"
Z95 = 1.959963984540054


def summarise(sample):
    sample = np.asarray(sample, dtype=float)
    standard_error = sample.std(ddof=1) / np.sqrt(sample.size)
    return {"mean": float(sample.mean()), "std": float(sample.std(ddof=1)),
            "median": float(np.median(sample)),
            "p5": float(np.percentile(sample, 5)),
            "p95": float(np.percentile(sample, 95)),
            "ci_low": float(sample.mean() - Z95 * standard_error),
            "ci_high": float(sample.mean() + Z95 * standard_error)}


def paired(sample, baseline):
    difference = np.asarray(sample, dtype=float) - np.asarray(baseline, dtype=float)
    out = summarise(difference)
    out["wins"] = float((difference > 0).mean())
    out["significant"] = bool(out["ci_low"] > 0.0 or out["ci_high"] < 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=int, default=config.N_EVAL)
    ap.add_argument("--tune", type=int, default=config.N_TUNE)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.eval, args.tune = 60, 40

    RESULTS.mkdir(exist_ok=True)
    log, started = [], time.time()

    def say(line):
        print(line)
        log.append(line)

    tune_seeds = simulator.scenario_seeds(config.TUNE_SEED, args.tune)
    eval_seeds = simulator.scenario_seeds(config.EVAL_SEED, args.eval)
    say(f"tuning on {args.tune} scenarios from seed {config.TUNE_SEED}, "
        f"reporting on {args.eval} from seed {config.EVAL_SEED}")
    say(f"venue {config.CAPACITY:,} seats, {config.SELLING_DAYS} days, "
        f"prices {config.PRICE_MIN:.0f} to {config.PRICE_MAX:.0f} in steps of "
        f"{config.PRICE_STEP:.0f}")

    # --- baseline: the best single fixed price, found by simulation ----------
    rows = []
    for price in config.PRICE_GRID:
        out = simulator.run_policy(policies.fixed_price(price), tune_seeds)
        rows.append({"price": price, "revenue": float(out["revenue"].mean()),
                     "sold": float(out["sold"].mean()),
                     "sold_out": float(out["sold_out"].mean())})
    fixed_table = pd.DataFrame(rows)
    fixed_table.to_csv(RESULTS / "tuning_fixed.csv", index=False)
    best_price = float(fixed_table.loc[fixed_table.revenue.idxmax(), "price"])
    say(f"\nbaseline            best fixed price {best_price:.0f} of "
        f"{len(config.PRICE_GRID)} tried")

    # --- PFA: the price schedule ---------------------------------------------
    rows = []
    for start, slope, scarcity in itertools.product(config.SCHEDULE_START_GRID,
                                                    config.SCHEDULE_SLOPE_GRID,
                                                    config.SCHEDULE_SCARCITY_GRID):
        out = simulator.run_policy(policies.price_schedule(start, slope, scarcity), tune_seeds)
        rows.append({"start": start, "slope": slope, "scarcity": scarcity,
                     "revenue": float(out["revenue"].mean()),
                     "sold": float(out["sold"].mean()),
                     "sold_out": float(out["sold_out"].mean())})
    schedule_table = pd.DataFrame(rows)
    schedule_table.to_csv(RESULTS / "tuning_schedule.csv", index=False)
    best = schedule_table.loc[schedule_table.revenue.idxmax()]
    start, slope, scarcity = float(best.start), float(best.slope), float(best.scarcity)
    say(f"PFA schedule        start {start:.0f}, slope {slope:.1f}, scarcity {scarcity:.1f}, "
        f"of {len(schedule_table)} combinations")

    # --- DLA: the lookahead horizon -------------------------------------------
    rows = []
    for horizon in config.LOOKAHEAD_HORIZON_GRID:
        out = simulator.run_policy(policies.lookahead(horizon), tune_seeds[:max(20, args.tune // 4)])
        rows.append({"horizon": horizon, "revenue": float(out["revenue"].mean()),
                     "sold": float(out["sold"].mean()),
                     "sold_out": float(out["sold_out"].mean())})
    lookahead_table = pd.DataFrame(rows)
    lookahead_table.to_csv(RESULTS / "tuning_lookahead.csv", index=False)
    horizon = int(lookahead_table.loc[lookahead_table.revenue.idxmax(), "horizon"])
    say(f"DLA lookahead       horizon {horizon} of {list(config.LOOKAHEAD_HORIZON_GRID)}, "
        f"tuned on a quarter of the scenarios because each day solves a programme")

    # --- the final comparison -------------------------------------------------
    specs = [
        (f"Fixed price of {best_price:.0f}", "baseline", policies.fixed_price(best_price)),
        (f"Schedule, start {start:.0f} slope {slope:.1f} scarcity {scarcity:.1f}", "PFA",
         policies.price_schedule(start, slope, scarcity)),
        (f"Lookahead over {horizon} days", "DLA", policies.lookahead(horizon)),
    ]
    runs, rows = {}, []
    for name, kind, make in specs:
        out = simulator.run_policy(make, eval_seeds)
        runs[name] = out
        revenue = summarise(out["revenue"])
        rows.append({"policy": name, "class": kind,
                     "revenue_mean": revenue["mean"], "ci_low": revenue["ci_low"],
                     "ci_high": revenue["ci_high"], "revenue_std": revenue["std"],
                     "revenue_p5": revenue["p5"], "revenue_p95": revenue["p95"],
                     "sold": float(out["sold"].mean()),
                     "sell_out_probability": float(out["sold_out"].mean()),
                     "average_price": float(out["average_price"].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "policies.csv", index=False)

    say("")
    say(f"{'policy':<48}{'class':<10}{'revenue':>11}{'5th':>11}{'95th':>11}"
        f"{'sold':>8}{'sellout':>9}{'avg p':>8}")
    for row in rows:
        say(f"{row['policy']:<48}{row['class']:<10}{row['revenue_mean']:>11,.0f}"
            f"{row['revenue_p5']:>11,.0f}{row['revenue_p95']:>11,.0f}"
            f"{row['sold']:>8,.0f}{row['sell_out_probability']:>8.0%}"
            f"{row['average_price']:>8.0f}")

    base_name = specs[0][0]
    rows = []
    for name, kind, make in specs[1:]:
        difference = paired(runs[name]["revenue"], runs[base_name]["revenue"])
        rows.append({"policy": name, "mean_difference": difference["mean"],
                     "ci_low": difference["ci_low"], "ci_high": difference["ci_high"],
                     "wins": difference["wins"], "significant": difference["significant"]})
    pd.DataFrame(rows).to_csv(RESULTS / "paired_differences.csv", index=False)
    say(f"\npaired against the baseline, same scenario seeds on both sides")
    for row in rows:
        say(f"  {row['policy']:<48}{row['mean_difference']:>10,.0f}"
            f"   [{row['ci_low']:,.0f}, {row['ci_high']:,.0f}]"
            f"   wins {row['wins']:.0%}"
            f"   {'significant' if row['significant'] else 'not significant'}")

    # --- how sensitive is this to the reference-price effect? -----------------
    rows = []
    for eta in (0.0, 0.3, 0.6, 1.0):
        for name, kind, make in specs:
            out = simulator.run_policy(make, eval_seeds, eta=eta)
            rows.append({"eta": eta, "policy": name,
                         "revenue": float(out["revenue"].mean()),
                         "sell_out_probability": float(out["sold_out"].mean())})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(RESULTS / "sensitivity_eta.csv", index=False)
    say("\nsensitivity to the penalty on pricing above the reference")
    pivot = sensitivity.pivot(index="policy", columns="eta", values="revenue")
    for name, line in pivot.iterrows():
        say(f"  {name:<48}" + "".join(f"{value:>11,.0f}" for value in line))

    np.save(RESULTS / "price_paths.npy",
            np.array([runs[name]["price_paths"][:100] for name, _, _ in specs]))
    say(f"\nran in {time.time() - started:.0f} s")
    (RESULTS / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
