"""Part 2: portfolio construction as a mixed-integer programme solved with Gurobi.

The objective is mean-variance in the utility form of deck "002 mean variance" page 6,

    max  w'mu - risk_penalty * w'Sigma w    subject to  w'1 = 1,

extended with the cardinality, minimum-lot and sector constraints the brief lists, which
are items 6.4, 6.6 and 6.11 of the course's MILP topic.  Mean-variance is chosen over
mean-CVaR, which the brief offers as the alternative, because mean-variance is the
objective the course develops and audits while CVaR appears in none of its material.

One model serves two purposes, which is deliberate.  With `w_prev` left out it builds the
Part 2 portfolio.  With `w_prev` and a trading cost supplied it is the single-period
rebalance of deck "007 Policy Classes" page 9, the one the deck calls a MILP with knobs,
and page 18 notes that the rolling version collapses to exactly this at horizon zero.
The deck's cash-buffer knob is not included, because the brief fixes the weights to sum
to one, so cash is held through the cash exchange traded fund in the universe instead.
"""
import numpy as np

from . import config

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except ImportError:                                   # pragma: no cover
    HAVE_GUROBI = False


def solve(mu, sigma, classes, risk_penalty: float, k_max: int = None,
          lo: float = None, up: float = None, cap: float = None,
          w_prev=None, cost: float = 0.0, time_limit: float = 30.0):
    """Choose weights w and selections z, and return the weights and a small report.

    `risk_penalty` is gamma / 2 when the portfolio is built for a client, and the tuned
    knob theta when the same model is used as a policy.  `classes` is one label per
    asset, and the sector constraint caps the total weight of each label.
    """
    if not HAVE_GUROBI:                               # pragma: no cover
        raise RuntimeError("gurobipy is required; install it with pip install -r requirements.txt")

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    k_max = config.K_MAX if k_max is None else k_max
    lo = config.MIN_POSITION if lo is None else lo
    up = config.MAX_POSITION if up is None else up
    cap = config.CLASS_CAP if cap is None else cap
    labels = sorted(set(classes))

    model = gp.Model("portfolio")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    w = model.addVars(n, lb=0.0, ub=up, name="w")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="budget")
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k_max, name="cardinality")
    for i in range(n):
        model.addConstr(w[i] >= lo * z[i], name=f"min_lot[{i}]")
        model.addConstr(w[i] <= up * z[i], name=f"max_lot[{i}]")
    for label in labels:
        members = [i for i in range(n) if classes[i] == label]
        model.addConstr(gp.quicksum(w[i] for i in members) <= cap, name=f"cap[{label}]")

    expected = gp.quicksum(mu[i] * w[i] for i in range(n))
    variance = gp.quicksum(sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(n))
    objective = expected - risk_penalty * variance

    turnover = None
    if w_prev is not None and cost > 0.0:
        w_prev = np.asarray(w_prev, dtype=float)
        t = model.addVars(n, lb=0.0, name="t")
        for i in range(n):
            model.addConstr(t[i] >= w[i] - w_prev[i], name=f"up[{i}]")
            model.addConstr(t[i] >= w_prev[i] - w[i], name=f"down[{i}]")
        turnover = gp.quicksum(t[i] for i in range(n))
        objective = objective - cost * turnover

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
        raise RuntimeError(f"the portfolio model returned status {model.Status}")

    weights = np.array([w[i].X for i in range(n)])
    weights[weights < 1e-9] = 0.0
    weights = weights / weights.sum()
    report = {
        "status": int(model.Status),
        "held": int((weights > 1e-9).sum()),
        "expected_monthly": float(weights @ mu),
        "monthly_vol": float(np.sqrt(weights @ sigma @ weights)),
        "objective": float(model.ObjVal),
        "turnover": None if turnover is None else float(turnover.getValue()),
        "gap": float(model.MIPGap) if model.SolCount else float("nan"),
    }
    return weights, report
