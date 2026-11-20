"""Command-line driver: tune the policies, evaluate them, and write every table.

    python run_analysis.py [--eval 500] [--tune 200] [--quick]

Writes CSVs to results/ and a plain-text digest to results/summary.txt.  The notebook
reads the same functions, so the report and this driver cannot disagree.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from bpd import config, data, evaluate, policies, portfolio, questionnaire, simulator  # noqa: E402

RESULTS = HERE / "results"


def build(shrink=None, risk_penalty=None):
    """The client, the universe and the Part 2 portfolio the policies start from."""
    rets = data.monthly_returns(data.load_prices())
    mu, sigma = data.moments(rets, shrink=shrink)
    classes = [config.ASSET_CLASS[t] for t in config.TICKERS]
    client = questionnaire.score(questionnaire.default_answers())
    penalty = client.gamma / 2.0 if risk_penalty is None else risk_penalty
    target, report = portfolio.solve(mu, sigma, classes, penalty)
    return rets, mu, sigma, classes, client, target, report


def policy_specs(target, mu, sigma, classes, band, penalty, every, n_assets):
    """Every policy in the comparison, with the class each one belongs to."""
    cash = config.TICKERS.index("BIL")
    equal = np.full(n_assets, 1.0 / n_assets)
    return [
        ("Buy and hold the Part 2 portfolio", "baseline",
         policies.buy_and_hold(target), target),
        ("Buy and hold equal weights", "baseline",
         policies.buy_and_hold(equal), equal),
        (f"Band rebalancing, band {band:.0%}", "PFA",
         policies.band_rebalance(target, band), target),
        (f"Re-solve every {every} months, penalty {penalty:g}", "CFA",
         policies.periodic_resolve(target, mu, sigma, classes, penalty, every), target),
        (f"Panic selling at {config.PANIC_DRAWDOWN:.0%}", "behavioural",
         policies.panic_selling(target, cash, config.PANIC_DRAWDOWN,
                                config.PANIC_RECOVERY), target),
        (f"Performance chasing, intensity {config.CHASE_INTENSITY:g}", "behavioural",
         policies.performance_chasing(target, config.CHASE_INTENSITY,
                                      config.CHASE_EVERY), target),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=int, default=config.N_EVAL)
    ap.add_argument("--tune", type=int, default=config.N_TUNE)
    ap.add_argument("--quick", action="store_true", help="small run for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.eval, args.tune = 60, 40

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    log, started = [], time.time()

    def say(line):
        print(line)
        log.append(line)

    if args.quick or args.eval < config.N_EVAL:
        say(f"SMOKE RUN on {args.eval} scenarios, which is fewer than the "
            f"{config.N_EVAL} the brief requires. These numbers are for checking that "
            f"the pipeline runs and must not be quoted anywhere.")
        say("")

    rets, mu, sigma, classes, client, target, build_report = build()
    R = rets.to_numpy()
    reference = simulator.reference_wealth(client.reference, config.INITIAL_WEALTH)

    say(f"months of history      {len(rets)}  ({rets.index[0].date()} to {rets.index[-1].date()})")
    say(f"client                 alpha {client.alpha:.3f}  beta {client.beta:.3f}  "
        f"lambda {client.lam:.3f}  gamma {client.gamma:.2f}")
    say(f"reference point        {client.reference}, {reference:,.0f}")
    say(f"Part 2 portfolio       {build_report['held']} holdings, "
        f"expected {build_report['expected_monthly'] * 12:.2%} a year, "
        f"volatility {build_report['monthly_vol'] * np.sqrt(12):.2%}")

    weights = pd.DataFrame({"ticker": config.TICKERS,
                            "asset_class": [config.ASSET_CLASS[t] for t in config.TICKERS],
                            "weight": target})
    weights[weights.weight > 1e-9].sort_values("weight", ascending=False).to_csv(
        RESULTS / "portfolio.csv", index=False)

    # --- tuning, on the tuning seed ------------------------------------------
    tune_paths = evaluate.scenarios(R, args.tune, config.TUNE_SEED)
    say(f"\ntuning on {args.tune} scenarios from seed {config.TUNE_SEED}")

    keeps_mandate = lambda out: evaluate.mandate_is_kept(out, config.MAX_POSITION)
    best_band, band_rows = evaluate.tune(
        lambda p: policies.band_rebalance(target, p[0]),
        [(b,) for b in config.BAND_GRID], tune_paths, target, client, reference,
        feasible=keeps_mandate)
    band = best_band["parameters"][0]
    unconstrained = max(band_rows, key=lambda row: row["score"])
    say(f"  PFA band               best {band:.0%} of {len(config.BAND_GRID)} values tried, "
        f"subject to keeping the {config.MAX_POSITION:.0%} position cap")
    say(f"                         without that constraint the search picks "
        f"{unconstrained['parameters'][0]:.0%}, which trades "
        f"{unconstrained['turnover_per_year']:.3f} times a year and ends with "
        f"{unconstrained['largest_end_weight_p95']:.0%} in one fund at the 95th percentile")
    pd.DataFrame([{"band": r["parameters"][0], **{k: v for k, v in r.items()
                                                  if k != "parameters"}}
                  for r in band_rows]).to_csv(RESULTS / "tuning_band.csv", index=False)
    say("    band    score      median W   turnover   95th pct largest weight   keeps mandate")
    for r in band_rows:
        say(f"    {r['parameters'][0]:>5.0%} {r['score']:>10,.0f} {r['median_terminal']:>13,.0f}"
            f" {r['turnover_per_year']:>10.3f} {r['largest_end_weight_p95']:>20.1%}"
            f"   {'yes' if r['feasible'] else 'no'}")

    cfa_grid = [(p, e) for p in config.RISK_PENALTY_GRID for e in config.RESOLVE_EVERY_GRID]
    best_cfa, cfa_rows = evaluate.tune(
        lambda p: policies.periodic_resolve(target, mu, sigma, classes, p[0], p[1]),
        cfa_grid, tune_paths, target, client, reference, feasible=keeps_mandate)
    penalty, every = best_cfa["parameters"]
    say(f"  CFA                    best penalty {penalty:g}, every {every} months, "
        f"of {len(cfa_grid)} combinations tried")
    pd.DataFrame([{"risk_penalty": r["parameters"][0], "every": r["parameters"][1],
                   **{k: v for k, v in r.items() if k != "parameters"}}
                  for r in cfa_rows]).to_csv(RESULTS / "tuning_cfa.csv", index=False)

    # --- final evaluation, on a different seed -------------------------------
    eval_paths = evaluate.scenarios(R, args.eval, config.EVAL_SEED)
    say(f"\nevaluating on {args.eval} scenarios from seed {config.EVAL_SEED}, "
        f"the same scenarios for every policy")

    specs = policy_specs(target, mu, sigma, classes, band, penalty, every, len(config.TICKERS))
    runs = {}
    rows = []
    for name, kind, make, start in specs:
        out = simulator.run_policy(make, eval_paths, start, client, reference)
        runs[name] = out
        value = evaluate.summarise(out["value"])
        terminal = evaluate.summarise(out["terminal"])
        rows.append({
            "policy": name, "class": kind,
            "value_mean": value["mean"], "value_ci_low": value["ci_low"],
            "value_ci_high": value["ci_high"], "value_std": value["std"],
            "terminal_mean": terminal["mean"], "terminal_median": terminal["median"],
            "terminal_p5": terminal["p5"], "terminal_p95": terminal["p95"],
            "terminal_std": terminal["std"],
            "shortfall": float((out["terminal"] < reference).mean()),
            "max_drawdown_mean": float(out["max_drawdown"].mean()),
            "turnover_per_year": float(out["turnover_per_year"].mean()),
            "months_traded": float(out["months_traded"].mean()),
            "cost_total_mean": float(out["cost_total"].mean()),
            "start_concentration": float((start ** 2).sum()),
            "end_concentration": float((out["terminal_weights"] ** 2).sum(1).mean()),
            "largest_end_weight": float(out["terminal_weights"].max(1).mean()),
            "largest_end_weight_worst": float(out["terminal_weights"].max()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "policies.csv", index=False)

    say("")
    say(f"{'policy':<42}{'class':<12}{'E[v]':>12}{'median W':>11}"
        f"{'5th':>10}{'95th':>10}{'short':>7}{'maxDD':>7}{'turn':>7}{'1/HHI':>7}{'max w':>7}")
    for row in rows:
        say(f"{row['policy']:<42}{row['class']:<12}{row['value_mean']:>12,.0f}"
            f"{row['terminal_median']:>11,.0f}{row['terminal_p5']:>10,.0f}"
            f"{row['terminal_p95']:>10,.0f}{row['shortfall']:>6.0%}"
            f"{row['max_drawdown_mean']:>7.1%}{row['turnover_per_year']:>7.2f}"
            f"{1.0 / row['end_concentration']:>7.1f}{row['largest_end_weight']:>6.0%}")

    # --- paired differences against the baseline -----------------------------
    base_name = specs[0][0]
    diff_rows = []
    for name in runs:
        if name == base_name:
            continue
        value = evaluate.paired_difference(runs[name]["value"], runs[base_name]["value"])
        terminal = evaluate.paired_difference(runs[name]["terminal"], runs[base_name]["terminal"])
        diff_rows.append({"policy": name, "value_diff_mean": value["mean"],
                          "value_ci_low": value["ci_low"], "value_ci_high": value["ci_high"],
                          "value_significant": value["significant"], "value_wins": value["wins"],
                          "terminal_diff_median": terminal["median"],
                          "terminal_diff_mean": terminal["mean"],
                          "terminal_ci_low": terminal["ci_low"],
                          "terminal_ci_high": terminal["ci_high"]})
    pd.DataFrame(diff_rows).to_csv(RESULTS / "paired_differences.csv", index=False)
    say("\npaired against the baseline, same scenarios on both sides")
    say(f"{'policy':<42}{'d E[v]':>13}{'95% CI':>26}{'wins':>7}{'d median W':>12}")
    for row in diff_rows:
        say(f"{row['policy']:<42}{row['value_diff_mean']:>13,.0f}"
            f"{'[' + format(row['value_ci_low'], ',.0f') + ', ' + format(row['value_ci_high'], ',.0f') + ']':>26}"
            f"{row['value_wins']:>6.0%}{row['terminal_diff_median']:>12,.0f}")

    # --- the cost of the client's behaviour, against the disciplined policy ---
    disciplined = specs[2][0]
    cost_rows = []
    for name in runs:
        if name == disciplined:
            continue
        terminal = evaluate.paired_difference(runs[name]["terminal"], runs[disciplined]["terminal"])
        cost_rows.append({"policy": name,
                          "median_terminal_vs_disciplined": terminal["median"],
                          "mean_terminal_vs_disciplined": terminal["mean"],
                          "ci_low": terminal["ci_low"], "ci_high": terminal["ci_high"],
                          "share_of_disciplined_median": float(
                              np.median(runs[name]["terminal"]) /
                              np.median(runs[disciplined]["terminal"]))})
    pd.DataFrame(cost_rows).to_csv(RESULTS / "behavioural_cost.csv", index=False)
    say(f"\ncost of behaviour against the disciplined policy, {disciplined}")
    for row in cost_rows:
        say(f"  {row['policy']:<42}{row['median_terminal_vs_disciplined']:>12,.0f}"
            f"   {row['share_of_disciplined_median']:>6.1%} of its median")

    # --- sensitivity ---------------------------------------------------------
    sens = []
    for multiple in config.TARGET_MULTIPLES:
        ref = config.INITIAL_WEALTH * multiple
        ranked = sorted(((float(np.mean(simulator.objective(
            runs[name]["terminal"], ref, client.alpha, client.beta, client.lam))), name)
            for name in runs), reverse=True)
        for rank, (score, name) in enumerate(ranked, start=1):
            sens.append({"setting": "target multiple", "value": multiple, "policy": name,
                         "value_mean": score, "rank": rank})
        say(f"\ntarget {multiple:g}x initial wealth, reference {ref:,.0f}: "
            f"best is {ranked[0][1]}")
    pd.DataFrame(sens).to_csv(RESULTS / "sensitivity_target.csv", index=False)

    panic_rows = []
    cash = config.TICKERS.index("BIL")
    for d in config.PANIC_DRAWDOWN_GRID:
        out = simulator.run_policy(
            policies.panic_selling(target, cash, d, config.PANIC_RECOVERY),
            eval_paths, target, client, reference)
        panic_rows.append({"drawdown": d,
                           "fires_in": float((out["months_traded"] > 0).mean()),
                           "median_terminal": float(np.median(out["terminal"])),
                           "value_mean": float(out["value"].mean()),
                           "max_drawdown_mean": float(out["max_drawdown"].mean())})
    pd.DataFrame(panic_rows).to_csv(RESULTS / "sensitivity_panic.csv", index=False)
    say("\npanic threshold sensitivity")
    for row in panic_rows:
        say(f"  drawdown {row['drawdown']:.0%}  fires in {row['fires_in']:>5.0%} of scenarios"
            f"   median W {row['median_terminal']:>11,.0f}"
            f"   mean maxDD {row['max_drawdown_mean']:.1%}")

    shrink_rows = []
    for shrink in (0.0, 0.25, 0.50, 0.75, 1.00):
        mu_s, sigma_s = data.moments(rets, shrink=shrink)
        w_s, rep_s = portfolio.solve(mu_s, sigma_s, classes, client.gamma / 2.0)
        out = simulator.run_policy(policies.buy_and_hold(w_s), eval_paths, w_s,
                                   client, reference)
        shrink_rows.append({"shrink": shrink, "held": rep_s["held"],
                            "expected_annual": rep_s["expected_monthly"] * 12,
                            "vol_annual": rep_s["monthly_vol"] * np.sqrt(12),
                            "median_terminal": float(np.median(out["terminal"])),
                            "value_mean": float(out["value"].mean()),
                            "largest_weight": float(w_s.max())})
    pd.DataFrame(shrink_rows).to_csv(RESULTS / "sensitivity_shrink.csv", index=False)
    say("\nshrinkage sensitivity, buy and hold on each resulting portfolio")
    for row in shrink_rows:
        say(f"  shrink {row['shrink']:.2f}  holdings {row['held']:>2d}"
            f"  largest weight {row['largest_weight']:>5.1%}"
            f"  volatility {row['vol_annual']:>5.1%}"
            f"  median W {row['median_terminal']:>11,.0f}")

    # --- the decisive robustness test: no asset carries a drift advantage ----
    equal_paths = evaluate.scenarios(data.equalise_means(R), args.eval, config.EVAL_SEED)
    spread = 12 * R.mean(axis=0).std()
    say("")
    say(f"with every asset given the same expected return, the cross-sectional spread of "
        f"{spread:.2%} a year removed")
    equal_runs, equal_rows = {}, []
    for name, kind, make, start in policy_specs(target, mu, sigma, classes, band,
                                                penalty, every, len(config.TICKERS)):
        out = simulator.run_policy(make, equal_paths, start, client, reference)
        equal_runs[name] = out
        equal_rows.append({"policy": name, "class": kind,
                           "value_mean": float(out["value"].mean()),
                           "median_terminal": float(np.median(out["terminal"])),
                           "max_drawdown_mean": float(out["max_drawdown"].mean())})
    for row in sorted(equal_rows, key=lambda r: -r["value_mean"]):
        say(f"  {row['policy']:<42}{row['value_mean']:>12,.0f}"
            f"{row['median_terminal']:>11,.0f}{row['max_drawdown_mean']:>8.1%}")
    equal_base = policy_specs(target, mu, sigma, classes, band, penalty, every,
                              len(config.TICKERS))[0][0]
    for row in equal_rows:
        if row["policy"] == equal_base:
            continue
        paired = evaluate.paired_difference(equal_runs[row["policy"]]["value"],
                                            equal_runs[equal_base]["value"])
        row["value_diff_vs_baseline"] = paired["mean"]
        row["ci_low"] = paired["ci_low"]
        row["ci_high"] = paired["ci_high"]
        row["significant"] = paired["significant"]
    pd.DataFrame(equal_rows).to_csv(RESULTS / "robustness_equal_means.csv", index=False)
    say("  paired against the same baseline on these scenarios")
    for row in equal_rows:
        if "ci_low" not in row:
            continue
        say(f"    {row['policy']:<40}{row['value_diff_vs_baseline']:>12,.0f}"
            f"   [{row['ci_low']:,.0f}, {row['ci_high']:,.0f}]"
            f"   {'significant' if row['significant'] else 'not significant'}")

    np.save(RESULTS / "wealth_paths_baseline.npy",
            runs[base_name]["wealth_paths"][:200].astype(np.float32))
    say(f"\nran in {time.time() - started:.0f} s")
    (RESULTS / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
