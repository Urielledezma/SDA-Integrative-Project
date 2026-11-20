# Project A — Behavioural portfolio dashboard

An advisory firm profiles a client, builds a portfolio for that profile, and then shows
the client what their own behaviour would do to it over ten years. The profile comes from
a ten-question instrument that estimates the three parameters of a prospect-theory value
function. The portfolio comes from a mixed-integer programme with cardinality, minimum lot
and asset-class constraints, solved with Gurobi. The ten years come from a block bootstrap
of monthly returns, in which six policies are compared on the same scenarios.

---

## Contents

| File | What it is |
|---|---|
| [`A_behavioral_portfolio.ipynb`](A_behavioral_portfolio.ipynb) | The written report, with every number computed by its own code cells |
| [`build_notebook.py`](build_notebook.py) | Assembles the report notebook from source. **The notebook's source of truth** |
| [`app.py`](app.py) | The dashboard the client drives |
| [`run_analysis.py`](run_analysis.py) | Command-line driver that writes every table without opening the notebook |
| [`fetch_data.py`](fetch_data.py) | Downloads the price history and caches it |
| [`src/bpd/`](src/bpd) | The library: configuration, data, profile, questionnaire, portfolio, simulator, policies, evaluation |
| [`tests/`](tests) | The properties the report's conclusions rest on |
| [`data/prices.csv`](data) | The cached price history, so the results reproduce without a network |
| [`results/`](results) | Tables as CSV and figures as PNG, regenerated on every run |

---

## Running it

From this folder:

```bash
streamlit run app.py          # the dashboard
python run_analysis.py        # every table into results/, about fifteen minutes
python run_analysis.py --quick  # smoke test with small samples, about a minute
python fetch_data.py          # refresh the cached prices, only if you want newer data

python build_notebook.py      # rebuild the report from source
jupyter nbconvert --to notebook --execute --inplace A_behavioral_portfolio.ipynb
```

From the repository root, `python -m pytest -q` runs this project's tests along with the
others.

---

## How the three parts fit together

**Part 1** turns ten answers into `alpha`, `beta` and `lambda`. The identification runs in
the order beta, alpha, lambda, because a loss-only lottery cancels loss aversion and
leaves curvature alone, and the mixed lottery then pins lambda given both curvatures. The
loss-aversion question is the opening exercise of the course's prospect-theory lecture and
the estimator is that lecture's own formula, so the tests check our implementation against
fifteen values printed on those slides.

**Part 2** turns the profile into a portfolio. Risk aversion is set from loss aversion, so
a client who needs a larger gain before accepting a symmetric bet is handed a more
defensive portfolio. The expected returns are pulled toward their cross-sectional average
before they reach the optimiser, deliberately and with the intensity reported, because the
course shows that a binding long-only constraint already shrinks the inputs silently and
that doing it on purpose is the better practice.

**Part 3** turns the portfolio into a decision problem. The state carries wealth, the
weights, the peak wealth a drawdown is measured against, and the value an untouched target
portfolio would have reached, which is what a client sitting in cash needs in order to
know when to return. The decision is the month's trade. The objective is the
prospect-theory value of terminal wealth against the client's target.

---

## What is ours rather than the course's

Three choices are ours, and the report says so in each place rather than presenting them
as established:

- the map from loss aversion to the mean-variance risk aversion used in Part 2;
- the intensity with which expected returns are shrunk, which the report varies from zero
  to one and reports throughout;
- the convention that the trading cost is charged on every peso that changes hands, while
  turnover is reported separately in the course's own definition of half that sum.

Everything else is either in the course material, which the report cites by deck and page,
or prescribed by the assignment itself.
