"""Print the figures the presenters have to know by heart.

    python presentation_figures.py

Three of them are deterministic and are computed here from the data and the profile, so
they are right whether or not the evaluation has been run. The rest come from the
five-hundred-scenario evaluation and are read out of results/, which means this refuses to
print them until that run exists and is the full one.

The point is that nobody retypes a number. A figure spoken from memory that the notebook
contradicts on screen is the worst thing that can happen in the room.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from bpd import config, data, portfolio, profile, questionnaire  # noqa: E402

RESULTS = HERE / "results"


def deterministic():
    rets = data.monthly_returns(data.load_prices())
    mu, sigma = data.moments(rets)
    classes = [config.ASSET_CLASS[t] for t in config.TICKERS]
    client = questionnaire.score(questionnaire.default_answers())
    answers = questionnaire.default_answers()
    gain = questionnaire._bracket_midpoint(answers["q1_ladder"])
    naive = profile.loss_aversion_from_g(gain, client.alpha, client.beta, corrected=False)
    weights, report = portfolio.solve(mu, sigma, classes, client.gamma / 2.0)
    spread = 12 * rets.to_numpy().mean(axis=0).std()

    print("Deterministic, right whether or not the evaluation has been run")
    print(f"  loss aversion, uncorrected      {naive:.2f}   the deck's body formula")
    print(f"  loss aversion, corrected        {client.lam:.2f}   the appendix's, "
          f"lower by {100 * (naive - client.lam) / naive:.1f}%")
    print(f"  risk aversion handed to Part 2  {client.gamma:.2f}")
    print(f"  cross-sectional spread removed  {spread:.2%} a year")
    print(f"  the portfolio                   {report['held']} holdings, "
          f"largest {weights.max():.1%} against a {config.MAX_POSITION:.0%} cap")
    print(f"  expected and volatility         {report['expected_monthly'] * 12:.2%} and "
          f"{report['monthly_vol'] * np.sqrt(12):.2%} a year")
    return client


def from_the_run(client):
    summary = RESULTS / "summary.txt"
    policies = RESULTS / "policies.csv"
    paired = RESULTS / "paired_differences.csv"
    if not (summary.exists() and policies.exists() and paired.exists()):
        print("\nFrom the evaluation: NOT AVAILABLE, results/ is incomplete.")
        print("  Run:  python run_analysis.py")
        return 1

    text = summary.read_text(encoding="utf-8")
    if "SMOKE RUN" in text:
        print("\nFrom the evaluation: NOT AVAILABLE, results/ holds a smoke run.")
        print("  Run the full one:  python run_analysis.py")
        return 1

    # The schema is not what makes a results folder stale. What makes it stale is that
    # the client it was computed for is no longer the client the code produces, which is
    # exactly what happened when the loss-aversion estimator was corrected. The driver
    # records the parameters it ran with, so they can be compared.
    import re
    match = re.search(r"lambda ([\d.]+)\s+gamma ([\d.]+)", text)
    if not match:
        print("\nFrom the evaluation: NOT AVAILABLE, summary.txt does not record which "
              "client it was run for, so it predates this check.")
        print("  Run:  python run_analysis.py")
        return 1
    ran_with = (float(match.group(1)), float(match.group(2)))
    now = (round(client.lam, 3), round(client.gamma, 2))
    if abs(ran_with[0] - now[0]) > 0.005 or abs(ran_with[1] - now[1]) > 0.02:
        print("\nFrom the evaluation: NOT AVAILABLE, results/ is stale.")
        print(f"  it was run for a client with lambda {ran_with[0]:.3f} and "
              f"gamma {ran_with[1]:.2f}")
        print(f"  the code now produces  lambda {now[0]:.3f} and gamma {now[1]:.2f}")
        print("  Run:  python run_analysis.py")
        return 1

    table = pd.read_csv(policies)
    differences = pd.read_csv(paired)

    # results/ can be left behind by an older run whose columns and policy names differ.
    # Saying so beats either crashing or printing a figure from a superseded evaluation.
    needed = {"policy", "class", "value_mean", "terminal_median", "largest_end_weight",
              "largest_end_weight_worst", "start_concentration", "end_concentration"}
    missing = needed - set(table.columns)
    baseline = table[table.policy.str.contains("Part 2 portfolio", case=False)]
    if missing or baseline.empty:
        print("\nFrom the evaluation: NOT AVAILABLE, results/ is from an older run.")
        if missing:
            print(f"  columns it does not have: {', '.join(sorted(missing))}")
        if baseline.empty:
            print(f"  policies it does have: {', '.join(table.policy.astype(str))}")
        print("  Run:  python run_analysis.py")
        return 1

    hold = baseline.iloc[0]
    best_held = table.sort_values("value_mean", ascending=False)
    disciplined = table[table["class"].isin(["PFA", "CFA"])].sort_values(
        "value_mean", ascending=False).iloc[0]
    match = differences[differences.policy == disciplined.policy]
    if match.empty:
        print("\nFrom the evaluation: NOT AVAILABLE, the two results files disagree about "
              "which policies were run.")
        print("  Run:  python run_analysis.py")
        return 1
    against = match.iloc[0]

    print("\nFrom the five-hundred-scenario evaluation")
    print(f"  median wealth, holding          {hold.terminal_median:,.0f}")
    print(f"  median wealth, recommended      {disciplined.terminal_median:,.0f}   "
          f"({disciplined.policy})")
    print(f"  what discipline costs           {against.value_diff_mean:,.0f}   "
          f"[{against.value_ci_low:,.0f}, {against.value_ci_high:,.0f}]")
    print(f"  largest holding at the end      {hold.largest_end_weight:.0%} on average, "
          f"{hold.largest_end_weight_worst:.0%} in the worst scenario")
    print(f"  effective holdings, start/end   "
          f"{1 / hold.start_concentration:.1f} to {1 / hold.end_concentration:.1f}")
    print(f"  highest scoring policy overall  {best_held.iloc[0].policy}")
    return 0


if __name__ == "__main__":
    sys.exit(from_the_run(deterministic()))
