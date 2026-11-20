"""The advisory dashboard, for a client who will not read the code.

    streamlit run app.py

The client answers the questionnaire, sees the profile it implies, the portfolio built for
that profile, the spread of wealth paths it produces over ten years, and what their own
stated tendencies would cost them.  Every number shown is computed by the same functions
the report uses. The page opens on its own defaults rather than on the parameters the
report tuned, so a number here matches the report only when the settings match, and the
sidebar shows what they are.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from bpd import (config, data, evaluate, policies, portfolio, profile,  # noqa: E402
                 questionnaire, simulator)

st.set_page_config(page_title="Behavioural portfolio dashboard", layout="wide")


@st.cache_data
def load_returns():
    rets = data.monthly_returns(data.load_prices())
    return rets.to_numpy(), list(rets.columns), str(rets.index[0].date()), str(rets.index[-1].date())


@st.cache_data
def build_portfolio(gamma, shrink, k_max, min_position, max_position, class_cap):
    rets = data.monthly_returns(data.load_prices())
    mu, sigma = data.moments(rets, shrink=shrink)
    classes = [config.ASSET_CLASS[t] for t in config.TICKERS]
    weights, report = portfolio.solve(mu, sigma, classes, gamma / 2.0, k_max=k_max,
                                      lo=min_position, up=max_position, cap=class_cap)
    return weights, report, mu, sigma


@st.cache_data
def make_scenarios(n_paths, horizon, seed):
    returns, _, _, _ = load_returns()
    return data.block_bootstrap(returns, n_paths, horizon, seed)


@st.cache_data
def run_one(policy_name, band, penalty, every, drawdown, recovery, intensity,
            n_paths, horizon, seed, initial, cost, reference, alpha, beta, lam,
            gamma, shrink, k_max, min_position, max_position, class_cap):
    """One policy on the shared scenarios, cached so the sliders stay responsive.

    Every argument is a plain number, so the cache key is the whole set of inputs and a
    change to any of them recomputes exactly what depends on it.
    """
    paths = make_scenarios(n_paths, horizon, seed)
    weights, _, mu, sigma = build_portfolio(gamma, shrink, k_max, min_position,
                                            max_position, class_cap)
    classes = [config.ASSET_CLASS[t] for t in config.TICKERS]
    cash = config.TICKERS.index("BIL")
    equal = np.full(len(config.TICKERS), 1.0 / len(config.TICKERS))
    makers = {
        "Hold, never trade": policies.buy_and_hold(weights),
        "Equal weights, never trade": policies.buy_and_hold(equal),
        "Rebalance on a band": policies.band_rebalance(weights, band),
        "Re-solve the optimiser": policies.periodic_resolve(weights, mu, sigma, classes,
                                                            penalty, every),
        "Sell out after a fall": policies.panic_selling(weights, cash, drawdown, recovery),
        "Chase last year's winners": policies.performance_chasing(weights, intensity),
    }
    start = equal if policy_name == "Equal weights, never trade" else weights
    client = questionnaire.ClientProfile(alpha=alpha, beta=beta, lam=lam, gamma=gamma,
                                         reference="")
    out = simulator.run_policy(makers[policy_name], paths, start, client, reference,
                               initial=initial, cost=cost)
    return {k: out[k] for k in ("terminal", "value", "max_drawdown", "turnover_per_year",
                                "months_traded", "wealth_paths", "terminal_weights")}


PLAIN_ENGLISH = {
    "Hold, never trade": "Buy the portfolio once and leave it alone. No costs, but the mix "
                         "drifts toward whatever grows fastest.",
    "Equal weights, never trade": "Put the same amount in all twenty five funds and leave it. "
                                  "The benchmark any method has to beat before it earns attention.",
    "Rebalance on a band": "Leave it alone while the mix stays close to the plan, and trade "
                           "back to the plan when any holding drifts past the band.",
    "Re-solve the optimiser": "Every few months, re-run the optimiser with today's holdings "
                              "and the cost of trading, and move only if it is worth paying for.",
    "Sell out after a fall": "What you said you would do: move everything to cash after a fall "
                             "and wait for the market to recover before going back in.",
    "Chase last year's winners": "What you said you would do: each year, shift money toward "
                                 "whatever has been winning.",
}
CLASS_OF = {
    "Hold, never trade": "baseline",
    "Equal weights, never trade": "baseline",
    "Rebalance on a band": "PFA",
    "Re-solve the optimiser": "CFA",
    "Sell out after a fall": "behavioural",
    "Chase last year's winners": "behavioural",
}

returns, tickers, first_month, last_month = load_returns()

st.title("Behavioural portfolio dashboard")
st.caption(f"Twenty five funds across five asset classes, monthly returns from {first_month} "
           f"to {last_month}. Wealth is simulated over ten years by resampling those months "
           f"in blocks of three to six.")

# --- the questionnaire -------------------------------------------------------
st.sidebar.header("Your answers")
answers = {}
defaults = questionnaire.default_answers()
with st.sidebar.form("questionnaire"):
    for item in questionnaire.QUESTIONS:
        if item["kind"] == "ladder":
            answers[item["id"]] = st.select_slider(
                item["text"], options=item["options"], value=defaults[item["id"]],
                format_func=lambda v: f"${v:,.0f}")
        elif item["kind"] == "amount":
            low, high = item["bounds"]
            answers[item["id"]] = st.slider(item["text"], float(low), float(high),
                                            float(defaults[item["id"]]), step=50.0)
        else:
            answers[item["id"]] = st.radio(item["text"], item["options"],
                                           index=item["options"].index(defaults[item["id"]]))
    submitted = st.form_submit_button("Update my profile")

st.sidebar.header("Your money")
initial = st.sidebar.number_input("Amount to invest", 10_000.0, 10_000_000.0,
                                  config.INITIAL_WEALTH, step=10_000.0)
target_multiple = st.sidebar.slider("Target, as a multiple of what you invest", 1.5, 6.0,
                                    config.HEADLINE_TARGET_MULTIPLE, step=0.5,
                                    help="Used only if question 5 says you compare against "
                                         "a target.")
n_paths = st.sidebar.select_slider("Scenarios to simulate", [100, 200, 500], value=200)

with st.sidebar.expander("Adviser settings"):
    shrink = st.slider("How far to pull expected returns toward their average", 0.0, 1.0,
                       config.MU_SHRINK, step=0.05)
    k_max = st.slider("Most holdings allowed", 3, 15, config.K_MAX)
    min_position = st.slider("Smallest position", 0.01, 0.10, config.MIN_POSITION, step=0.01)
    max_position = st.slider("Largest position", 0.10, 0.60, config.MAX_POSITION, step=0.05)
    class_cap = st.slider("Most in any one asset class", 0.20, 1.00, config.CLASS_CAP, step=0.05)
    cost = st.slider("Trading cost, basis points", 0.0, 50.0,
                     config.COST_PER_TRADE * 10_000, step=1.0) / 10_000.0
    band = st.slider("Rebalancing band", 0.01, 0.30, 0.05, step=0.01)
    penalty = st.select_slider("Optimiser risk penalty", list(config.RISK_PENALTY_GRID), value=3.0)
    every = st.select_slider("Months between re-solves", list(config.RESOLVE_EVERY_GRID), value=12)
    drawdown = st.slider("Fall that makes you sell", 0.05, 0.40, config.PANIC_DRAWDOWN, step=0.05)
    recovery = st.slider("Recovery before you buy back", 0.02, 0.30, config.PANIC_RECOVERY,
                         step=0.02)
    intensity = st.slider("How hard you chase winners", 0.0, 2.0, config.CHASE_INTENSITY, step=0.25)

client = questionnaire.score(answers)
reference = simulator.reference_wealth(client.reference, initial, initial * target_multiple)
weights, report, mu, sigma = build_portfolio(client.gamma, shrink, k_max, min_position,
                                             max_position, class_cap)
SETTINGS = (client.gamma, shrink, k_max, min_position, max_position, class_cap)

# --- the profile -------------------------------------------------------------
st.header("What your answers say about you")
a, b, c, d = st.columns(4)
a.metric("Loss aversion", f"{client.lam:.2f}",
         help="How much larger a gain has to be than a loss before you accept a coin flip. "
              "Published estimates run from about 1.4 to 3.0.")
b.metric("Curvature over gains", f"{client.alpha:.2f}",
         help="Below one means each extra peso of gain is worth a little less than the last.")
c.metric("Curvature over losses", f"{client.beta:.2f}")
d.metric("Risk aversion used", f"{client.gamma:.1f}",
         help="Set from your loss aversion, and it is what decides how defensive the "
              "portfolio below is.")

left, right = st.columns([1, 1])
with left:
    flags = [name.replace("_", " ") for name, on in client.flags.items() if on]
    if flags:
        st.warning("Tendencies your answers point to: " + ", ".join(flags) +
                   ". The last two policies below put numbers on what they cost.")
    else:
        st.success("Your answers point to no particular tendency, so the behavioural "
                   "policies below describe someone else rather than you.")
    for note in client.notes:
        st.info(note)
with right:
    grid = np.linspace(-0.5 * initial, 0.5 * initial, 400)
    figure, axis = plt.subplots(figsize=(5, 2.6))
    axis.plot(grid, profile.value(grid, client.alpha, client.beta, client.lam))
    axis.axhline(0.0, linewidth=0.6, color="black")
    axis.axvline(0.0, linewidth=0.6, color="black")
    axis.set_xlabel("gain or loss against your target")
    axis.set_ylabel("value to you")
    axis.set_title("Your value function")
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

# --- the portfolio -----------------------------------------------------------
st.header("The portfolio built for that profile")
left, right = st.columns([1, 1])
with left:
    held = pd.DataFrame({"fund": config.TICKERS,
                         "asset class": [config.ASSET_CLASS[t] for t in config.TICKERS],
                         "weight": weights})
    held = held[held.weight > 1e-9].sort_values("weight", ascending=False)
    st.dataframe(held.style.format({"weight": "{:.1%}"}), hide_index=True,
                 use_container_width=True)
with right:
    st.metric("Expected return a year", f"{report['expected_monthly'] * 12:.1%}")
    st.metric("Volatility a year", f"{report['monthly_vol'] * np.sqrt(12):.1%}")
    st.metric("Holdings", f"{report['held']} of at most {k_max}")
    st.caption("Chosen by a mixed-integer programme: at most the number of holdings above, "
               "nothing smaller than the minimum position, nothing larger than the maximum, "
               "and a cap on each asset class. Expected returns are pulled toward their "
               "average first, because a mean estimated from ten years of data is not "
               "precise enough to be taken at face value.")

# --- the simulation ----------------------------------------------------------
st.header("What could happen over ten years")
chosen = st.multiselect("Policies to compare", list(PLAIN_ENGLISH),
                        default=["Hold, never trade", "Rebalance on a band",
                                 "Sell out after a fall", "Chase last year's winners"])
if not chosen:
    st.stop()

outs = {}
progress = st.progress(0.0, text="simulating")
for i, name in enumerate(chosen):
    outs[name] = run_one(name, band, penalty, every, drawdown, recovery,
                         intensity, n_paths, config.HORIZON_MONTHS, config.EVAL_SEED,
                         initial, cost, reference, client.alpha, client.beta, client.lam,
                         *SETTINGS)
    progress.progress((i + 1) / len(chosen), text=f"simulating {name}")
progress.empty()

fan_of = st.selectbox("Show the spread of paths for", chosen)
paths = outs[fan_of]["wealth_paths"]
months = np.arange(paths.shape[1])
figure, axis = plt.subplots(figsize=(10, 4))
axis.fill_between(months / 12, np.percentile(paths, 5, axis=0),
                  np.percentile(paths, 95, axis=0), alpha=0.25, color="grey",
                  label="5th to 95th")
axis.fill_between(months / 12, np.percentile(paths, 25, axis=0),
                  np.percentile(paths, 75, axis=0), alpha=0.45, color="grey",
                  label="25th to 75th")
axis.plot(months / 12, np.median(paths, axis=0), color="black", label="median")
axis.axhline(reference, linestyle="--", linewidth=1.0, color="black",
             label=f"your target, {reference:,.0f}")
axis.set_xlabel("years")
axis.set_ylabel("wealth")
axis.set_title(f"{fan_of}: {n_paths} scenarios")
axis.legend(loc="upper left", fontsize=8)
st.pyplot(figure, use_container_width=True)
plt.close(figure)

# --- the comparison ----------------------------------------------------------
st.header("The policies side by side")
DISCIPLINED = "Rebalance on a band"
baseline = DISCIPLINED if DISCIPLINED in outs else None
rows = []
for name, out in outs.items():
    terminal = out["terminal"]
    value = evaluate.summarise(out["value"])
    rows.append({
        "policy": name, "class": CLASS_OF[name],
        "median wealth": float(np.median(terminal)),
        "5th": float(np.percentile(terminal, 5)),
        "95th": float(np.percentile(terminal, 95)),
        "chance of missing the target": float((terminal < reference).mean()),
        "worst fall along the way": float(out["max_drawdown"].mean()),
        "trades a year": float(out["months_traded"].mean() / (config.HORIZON_MONTHS / 12)),
        "value to you": value["mean"],
    })
table = pd.DataFrame(rows).sort_values("value to you", ascending=False)
st.dataframe(table.style.format({
    "median wealth": "{:,.0f}", "5th": "{:,.0f}", "95th": "{:,.0f}",
    "chance of missing the target": "{:.0%}", "worst fall along the way": "{:.1%}",
    "trades a year": "{:.2f}", "value to you": "{:,.0f}"}), hide_index=True,
    use_container_width=True)

for name in chosen:
    st.markdown(f"**{name}** ({CLASS_OF[name]}). {PLAIN_ENGLISH[name]}")

# --- what the tendencies cost ------------------------------------------------
st.header("What your tendencies would cost you")
if baseline is not None:
    disciplined = np.median(outs[baseline]["terminal"])
    cost_rows = []
    for name in ("Sell out after a fall", "Chase last year's winners"):
        if name not in outs:
            continue
        difference = evaluate.paired_difference(outs[name]["terminal"], outs[baseline]["terminal"])
        cost_rows.append({"tendency": name,
                          "median wealth": float(np.median(outs[name]["terminal"])),
                          "against the disciplined policy": difference["median"],
                          "as a share of it": float(np.median(outs[name]["terminal"]) / disciplined)})
    if cost_rows:
        for row in cost_rows:
            delta = row["against the disciplined policy"]
            st.metric(row["tendency"], f"{row['median wealth']:,.0f}",
                      delta=f"{delta:,.0f} against staying disciplined")
        st.caption(f"Measured scenario by scenario against {baseline.lower()}, on the same "
                   f"{n_paths} scenarios, so the difference is the policy and not the luck of "
                   f"the draw.")
    else:
        st.info("Add one of the two behavioural policies above to see what it would cost.")
else:
    st.info("The cost of a tendency is measured against disciplined rebalancing, so add "
            "**Rebalance on a band** to the comparison above to see it. Comparing a "
            "tendency against another tendency would not say what it costs.")

# --- the recommendation ------------------------------------------------------
st.header("What we would do")
best = table.iloc[0]
hold_name = "Hold, never trade" if "Hold, never trade" in outs else chosen[0]
drifted = outs[hold_name]["terminal_weights"]
st.markdown(
    f"On your answers and your target of {reference:,.0f}, the policy that scores highest on "
    f"your own value function is **{best['policy'].lower()}**, at a median of "
    f"{best['median wealth']:,.0f} with a {best['chance of missing the target']:.0%} chance of "
    f"missing the target.")
kept = {row["policy"]: float(np.percentile(outs[row["policy"]]["terminal_weights"].max(1), 95))
        for _, row in table.iterrows()}
admissible = [name for name, largest in kept.items() if largest <= max_position]

if admissible:
    recommended = next(row["policy"] for _, row in table.iterrows()
                       if row["policy"] in admissible)
else:
    recommended = best["policy"]

if recommended == best["policy"]:
    st.markdown(
        f"That is also what we would recommend, because it keeps your largest holding at "
        f"{kept[recommended]:.0%} in nineteen scenarios out of twenty, inside the "
        f"{max_position:.0%} limit the portfolio was built under.")
else:
    st.markdown(
        f"We would still recommend **{recommended.lower()}** instead, because the "
        f"highest-scoring policy abandons the plan on the way. Measured on "
        f"{hold_name.lower()}, over ten years the largest holding grows from "
        f"{weights.max():.0%} of the portfolio to {drifted.max(1).mean():.0%} on average and "
        f"{drifted.max():.0%} in the worst scenario, which is not the risk you were profiled "
        f"for, while {recommended.lower()} keeps it at {kept[recommended]:.0%} in nineteen "
        f"scenarios out of twenty.")

if not admissible:
    st.warning(
        f"On these settings no policy shown keeps the largest holding inside {max_position:.0%} "
        f"at the 95th percentile, so the recommendation above is the highest scoring one rather "
        f"than one that holds the plan. Narrowing the rebalancing band usually fixes this.")
