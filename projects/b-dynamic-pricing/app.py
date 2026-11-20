"""The promoter's dashboard, for someone who will not read the code.

    streamlit run app.py

The promoter sets the venue, the selling window, the price bounds and what they believe
about demand, and sees the price path each policy would follow, the revenue it would
earn, how often it sells out, and how much of the answer depends on buyers comparing
today's price with what they remember.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from dtp import config, policies, simulator  # noqa: E402

st.set_page_config(page_title="Dynamic ticket pricing", layout="wide")

PLAIN_ENGLISH = {
    "One fixed price": "Pick a price before the tickets go on sale and never change it. "
                       "Simple to explain, and the bar anything else has to beat.",
    "A published schedule": "Announce a price that comes down as the date approaches, and "
                            "nudge it up or down depending on whether sales are ahead of "
                            "or behind where they should be.",
    "Re-plan every day": "Each morning, work out the best price for every remaining day "
                         "given the seats left, then charge only today's and do it again "
                         "tomorrow.",
}
CLASS_OF = {"One fixed price": "baseline", "A published schedule": "PFA",
            "Re-plan every day": "DLA"}


@st.cache_data
def run_one(name, price, start, slope, scarcity, horizon, n_scenarios, seed,
            capacity, days, arrival_a, arrival_b, wtp_median, wtp_log_sd,
            theta, eta, gamma, kappa):
    seeds = simulator.scenario_seeds(seed, n_scenarios)
    makers = {"One fixed price": policies.fixed_price(price),
              "A published schedule": policies.price_schedule(start, slope, scarcity),
              "Re-plan every day": policies.lookahead(horizon, arrival_a, arrival_b)}
    out = simulator.run_policy(makers[name], seeds, capacity=capacity, days=days,
                               arrival_a=arrival_a, arrival_b=arrival_b,
                               wtp_median=wtp_median, wtp_log_sd=wtp_log_sd,
                               theta=theta, eta=eta, gamma=gamma, kappa=kappa)
    return {k: out[k] for k in ("revenue", "sold", "sold_out", "average_price",
                                "price_paths", "seats_paths")}


@st.cache_data
def best_fixed_price(n_scenarios, seed, capacity, days, arrival_a, arrival_b,
                     wtp_median, wtp_log_sd, theta, eta, gamma, kappa):
    seeds = simulator.scenario_seeds(seed, max(40, n_scenarios // 4))
    rows = []
    for price in config.PRICE_GRID:
        out = simulator.run_policy(policies.fixed_price(price), seeds, capacity=capacity,
                                   days=days, arrival_a=arrival_a, arrival_b=arrival_b,
                                   wtp_median=wtp_median, wtp_log_sd=wtp_log_sd,
                                   theta=theta, eta=eta, gamma=gamma, kappa=kappa)
        rows.append({"price": price, "revenue": float(out["revenue"].mean())})
    table = pd.DataFrame(rows)
    return float(table.loc[table.revenue.idxmax(), "price"]), table


st.title("Dynamic ticket pricing")
st.caption("Three ways to price a run of tickets, compared on the same simulated buyers. "
           "The demand model is the one in the brief: buyers arrive at a rate that rises "
           "toward the event, each carries a price they would pay, and they resent paying "
           "more than the price they remember.")

st.sidebar.header("The event")
capacity = st.sidebar.number_input("Seats", 100, 20_000, config.CAPACITY, step=100)
days = st.sidebar.slider("Days on sale", 10, 120, config.SELLING_DAYS, step=5)
n_scenarios = st.sidebar.select_slider("Scenarios to simulate", [50, 100, 200, 500],
                                       value=100)

st.sidebar.header("What you believe about demand")
wtp_median = st.sidebar.slider("The price half your buyers would pay", 40.0, 200.0,
                               config.WTP_MEDIAN, step=5.0)
wtp_log_sd = st.sidebar.slider("How much buyers differ", 0.2, 1.2, config.WTP_LOG_SD,
                               step=0.05)
arrival_b = st.sidebar.slider("How much interest builds toward the date", 0.0, 3.0,
                              config.ARRIVAL_B, step=0.1)
arrival_a = st.sidebar.slider("Interest on the first day, in buyers", 1.0, 80.0,
                              config.ARRIVAL_A, step=1.0)

st.sidebar.header("Memory of past prices")
eta = st.sidebar.slider("How much they resent a price above what they remember", 0.0, 1.5,
                        config.ETA, step=0.1)
gamma = st.sidebar.slider("How much they reward a price below it", 0.0, 1.0,
                          config.GAMMA, step=0.05)
theta = st.sidebar.slider("How long they remember", 0.0, 0.95, config.THETA, step=0.05)
kappa = st.sidebar.slider("Rush when seats run low", 1.0, 3.0, config.KAPPA, step=0.1)

with st.sidebar.expander("Policy settings"):
    start = st.slider("Schedule: opening price", 60.0, 200.0, 120.0, step=10.0)
    slope = st.slider("Schedule: how far it falls", 0.0, 1.0, 0.6, step=0.1)
    scarcity = st.slider("Schedule: reaction to sales", 0.0, 0.6, 0.4, step=0.1)
    horizon = st.select_slider("Re-plan: days looked ahead",
                               list(config.LOOKAHEAD_HORIZON_GRID), value=10)

settings = (capacity, days, arrival_a, arrival_b, wtp_median, wtp_log_sd,
            theta, eta, gamma, kappa)

expected = sum(simulator.arrival_rate(t / max(1, days - 1), arrival_a, arrival_b)
               for t in range(days))
st.info(f"On these beliefs about {expected:,.0f} buyers look at the tickets over "
        f"{days} days, against {capacity:,} seats, so demand is "
        f"{expected / capacity:.1f} times capacity. Below about one, no policy sells out "
        f"and the question is only what price to charge.")

price, fixed_table = best_fixed_price(n_scenarios, config.EVAL_SEED, *settings)
st.caption(f"The best single price on these beliefs is {price:,.0f}, found by trying all "
           f"{len(config.PRICE_GRID)} of them. That is the baseline.")

chosen = st.multiselect("Policies to compare", list(PLAIN_ENGLISH),
                        default=list(PLAIN_ENGLISH))
if not chosen:
    st.stop()

outs = {}
progress = st.progress(0.0, text="simulating")
for i, name in enumerate(chosen):
    outs[name] = run_one(name, price, start, slope, scarcity, horizon, n_scenarios,
                         config.EVAL_SEED, *settings)
    progress.progress((i + 1) / len(chosen), text=f"simulating {name.lower()}")
progress.empty()

st.header("What each policy earns")
rows = []
for name in chosen:
    revenue = outs[name]["revenue"]
    standard_error = revenue.std(ddof=1) / np.sqrt(revenue.size)
    rows.append({"policy": name, "class": CLASS_OF[name],
                 "revenue": float(revenue.mean()),
                 "worst 5%": float(np.percentile(revenue, 5)),
                 "best 5%": float(np.percentile(revenue, 95)),
                 "chance of selling out": float(outs[name]["sold_out"].mean()),
                 "tickets sold": float(outs[name]["sold"].mean()),
                 "average price paid": float(outs[name]["average_price"].mean()),
                 "give or take": float(1.96 * standard_error)})
table = pd.DataFrame(rows).sort_values("revenue", ascending=False)
st.dataframe(table.style.format({
    "revenue": "{:,.0f}", "worst 5%": "{:,.0f}", "best 5%": "{:,.0f}",
    "chance of selling out": "{:.0%}", "tickets sold": "{:,.0f}",
    "average price paid": "{:,.0f}", "give or take": "±{:,.0f}"}),
    hide_index=True, use_container_width=True)

baseline = "One fixed price"
if baseline in outs:
    st.subheader("Against one fixed price, on the same buyers")
    for name in chosen:
        if name == baseline:
            continue
        difference = outs[name]["revenue"] - outs[baseline]["revenue"]
        standard_error = difference.std(ddof=1) / np.sqrt(difference.size)
        low = difference.mean() - 1.96 * standard_error
        high = difference.mean() + 1.96 * standard_error
        verdict = "better" if low > 0 else ("worse" if high < 0 else "too close to call")
        st.metric(name, f"{difference.mean():+,.0f}",
                  delta=f"{verdict}, between {low:+,.0f} and {high:+,.0f}")
    st.caption("Every policy meets the same simulated buyers, so these differences are "
               "the policy rather than the luck of the draw.")

st.header("The price path each one follows")
figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
grid = np.arange(days)
for name in chosen:
    paths = outs[name]["price_paths"]
    left.plot(grid, np.median(paths, axis=0), label=name)
    seats = outs[name]["seats_paths"]
    right.plot(np.arange(days + 1), np.median(seats, axis=0), label=name)
left.set_xlabel("days on sale")
left.set_ylabel("price")
left.set_title("Median price charged")
left.legend(fontsize=8)
right.set_xlabel("days on sale")
right.set_ylabel("seats left")
right.set_title("Median seats remaining")
right.axhline(0, linewidth=0.6, color="black")
right.legend(fontsize=8)
plt.tight_layout()
st.pyplot(figure, use_container_width=True)
plt.close(figure)

st.header("How much of this rests on buyers remembering prices")
st.caption("The demand model says a buyer resents a price above the one they remember and "
           "is tempted by one below it. If that effect is weaker or stronger than assumed, "
           "the ranking can move, so it is worth knowing how much.")
rows = []
for test_eta in (0.0, 0.3, 0.6, 1.0):
    for name in chosen:
        out = run_one(name, price, start, slope, scarcity, horizon,
                      max(50, n_scenarios // 2), config.EVAL_SEED, capacity, days,
                      arrival_a, arrival_b, wtp_median, wtp_log_sd, theta, test_eta,
                      gamma, kappa)
        rows.append({"resentment": test_eta, "policy": name,
                     "revenue": float(out["revenue"].mean())})
pivot = pd.DataFrame(rows).pivot(index="policy", columns="resentment", values="revenue")
st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)

st.header("What we would do")
winner = table.iloc[0]
runner_up = table.iloc[1] if len(table) > 1 else None
margin = None if runner_up is None else winner["revenue"] - runner_up["revenue"]
decisive = margin is not None and margin > winner["give or take"] + runner_up["give or take"]

st.markdown(
    f"On these beliefs, **{winner['policy'].lower()}** earns the most, at "
    f"{winner['revenue']:,.0f} give or take {winner['give or take']:,.0f}, selling out in "
    f"{winner['chance of selling out']:.0%} of scenarios at an average of "
    f"{winner['average price paid']:,.0f} a ticket.")
if runner_up is not None:
    st.markdown(
        f"It is ahead of {runner_up['policy'].lower()} by {margin:,.0f}, which "
        + ("is more than the uncertainty in either figure, so the ranking is firm."
           if decisive else
           "is inside the uncertainty in those figures, so treat the two as equal and "
           "pick on whichever is easier to explain to buyers."))
st.markdown(
    "Two things to weigh that no number here captures. A policy that changes the price "
    "every day is harder to announce and can annoy the buyers who went first, which the "
    "resentment slider only partly represents. And every figure on this page assumes the "
    "demand model is right, so the sliders are the honest part of the answer.")
