"""Part 3: the sequential decision problem, written with the five elements.

Step 3 of the course's six-step process, in the notation of deck "005 SDA" page 15.

State
    S_t = (W_t, w_t, M_t, I_t, H_t)
    W_t   wealth in pesos at the start of month t, a physical state
    w_t   portfolio weights, dimensionless, non-negative and summing to one
    M_t   the highest wealth reached up to month t, in pesos, which is what a
          drawdown is measured against
    I_t   the value of the target portfolio held without trading, in pesos, an
          informational state carried because a client sitting in cash needs the
          market rather than their own wealth to tell them when to return
    H_t   the last twelve months of asset returns, dimensionless, which is what a
          performance-chasing rule reads

    A policy that needs more memory than this, such as the panic rule's record of
    whether the client is currently in cash and how low the market went since, keeps
    it inside itself rather than in S_t, and `run_policy` builds one policy per
    scenario so that memory never crosses from one scenario to the next.

Decision
    x_t = w_t^+ - w_t, the change in weights executed at the start of month t, so the
    post-trade weights are w_t^+.  Everything in S_t is known when x_t is chosen and the
    return of month t+1 is not, which is the ordering deck 005 page 6 insists on.

Exogenous information
    W_{t+1} = r_{t+1}, the vector of asset returns realised during month t+1, supplied by
    the block bootstrap of the uncertainty model.

Transition function
    cost_t   = c * sum_i |x_{t,i}| * W_t          with c = 10 basis points
    W_{t+1}  = (W_t - cost_t) * (1 + w_t^+ . r_{t+1})
    w_{t+1}  = w_t^+ * (1 + r_{t+1}) / (1 + w_t^+ . r_{t+1})
    M_{t+1}  = max(M_t, W_{t+1})
    I_{t+1}  = I_t * (1 + w_target . r_{t+1})

    Every peso that changes hands pays the cost, so the charge is levied on sum_i |x_i|
    rather than on the half of it that deck "002 mean variance" page 23 calls turnover.
    That convention is OURS; turnover in the deck's definition, one half of that sum, is
    reported separately so the two are never confused.

Objective
    max over pi of E[ v(W_T - rho) ], with v the client's prospect-theory value function
    and rho the reference point from question 5 of the questionnaire.  The mean and the
    percentiles of terminal wealth are reported next to it, because the brief asks for
    both and because a value in prospect-theory units is not money.
"""
import numpy as np

from . import config, profile


def reference_wealth(reference: str, initial: float, target: float = None) -> float:
    """The rho of the objective, in dollars.

    Deck "003 Prospect Theory" page 24 calls the reference point an input rather than an
    output, and page 19 lists the candidates a client might use, so it is asked in
    question 5 rather than estimated.
    """
    if reference == "A target I have in mind":
        if target is not None:
            return float(target)
        return float(initial * config.HEADLINE_TARGET_MULTIPLE)
    return float(initial)


def simulate_path(policy, returns: np.ndarray, target: np.ndarray,
                  initial: float = None, cost: float = None):
    """Run one scenario month by month and return what the metrics need.

    `policy` is called as policy(t, state) and returns the post-trade weights, so a
    policy that returns the weights it was given does not trade.  A policy that needs to
    remember something, such as whether the client is currently sitting in cash, keeps
    that memory inside itself and is therefore built fresh for each scenario by
    `run_policy`.  The cost of trading and the transition are applied here rather than
    inside any policy, which keeps every policy comparable on the same accounting.
    """
    initial = config.INITIAL_WEALTH if initial is None else initial
    cost = config.COST_PER_TRADE if cost is None else cost
    horizon = returns.shape[0]
    target = np.asarray(target, dtype=float)

    wealth = float(initial)
    weights = target.copy()
    peak = wealth
    index = float(initial)

    wealth_path = np.empty(horizon + 1)
    wealth_path[0] = wealth
    traded_total = 0.0
    costs_total = 0.0
    trades = 0

    for t in range(horizon):
        state = {"wealth": wealth, "weights": weights, "peak": peak, "index": index,
                 "history": returns[max(0, t - 12):t], "target": target}

        desired = np.asarray(policy(t, state), dtype=float)
        traded = float(np.abs(desired - weights).sum())
        if traded > 1e-10:
            charge = cost * traded * wealth
            wealth -= charge
            costs_total += charge
            traded_total += traded
            trades += 1
            weights = desired

        r = returns[t]
        growth = 1.0 + float(weights @ r)
        wealth *= growth
        weights = weights * (1.0 + r) / growth
        peak = max(peak, wealth)
        index *= 1.0 + float(target @ r)
        wealth_path[t + 1] = wealth

    running_peak = np.maximum.accumulate(wealth_path)
    drawdown = 1.0 - wealth_path / running_peak
    return {
        "terminal": wealth,
        "terminal_weights": weights,
        "wealth_path": wealth_path,
        "max_drawdown": float(drawdown.max()),
        "turnover_per_year": float(0.5 * traded_total / (horizon / 12.0)),
        "cost_total": costs_total,
        "months_traded": trades,
    }


def objective(terminal, reference: float, alpha: float, beta: float, lam: float):
    """The prospect-theory value of terminal wealth relative to the reference point."""
    return profile.value(np.asarray(terminal, dtype=float) - reference, alpha, beta, lam)


def run_policy(make_policy, paths: np.ndarray, target: np.ndarray, client,
               reference: float, initial: float = None, cost: float = None):
    """Apply one policy to every scenario and collect the metrics for all of them.

    `make_policy` is a factory rather than a policy, so each scenario gets a policy with
    its own memory and nothing leaks from one scenario into the next.
    """
    runs = [simulate_path(make_policy(), paths[i], target, initial, cost)
            for i in range(len(paths))]
    terminal = np.array([run["terminal"] for run in runs])
    return {
        "terminal": terminal,
        "value": objective(terminal, reference, client.alpha, client.beta, client.lam),
        "max_drawdown": np.array([run["max_drawdown"] for run in runs]),
        "turnover_per_year": np.array([run["turnover_per_year"] for run in runs]),
        "cost_total": np.array([run["cost_total"] for run in runs]),
        "months_traded": np.array([run["months_traded"] for run in runs]),
        "wealth_paths": np.array([run["wealth_path"] for run in runs]),
        "terminal_weights": np.array([run["terminal_weights"] for run in runs]),
    }
