"""The demand model and the sequential decision problem, written with the five elements.

The brief supplies this model as equations rather than as code, so what follows is a
transcription of it and every departure is marked.

State
    S_t = (R_t, d_t, r_t, H_t)
    R_t   seats remaining at the start of day t, an integer count
    d_t   days left until the event, an integer, so d_t = T - t
    r_t   the reference price buyers carry, in pesos
    H_t   sales over the last few days, in tickets, which a price-schedule rule reads

Decision
    x_t = p_t, the price charged on day t, taken from the grid.  Everything in S_t is
    known when it is chosen and the day's arrivals are not.

Exogenous information
    W_{t+1} = (N_{t+1}, V_{t+1}), the number of buyers who arrive during day t and the
    willingness to pay each of them carries.

Transition function
    N_{t+1} ~ Poisson(lambda(u_t) * kappa if R_t < 0.10 C else lambda(u_t))
    buy_i   = 1 if V_i >= p_t + eta max(p_t - r_t, 0) - gamma max(r_t - p_t, 0)
    S_{t+1} = min(R_t, sum_i buy_i)          tickets actually sold
    R_{t+1} = R_t - S_{t+1}
    r_{t+1} = theta r_t + (1 - theta) p_t
    d_{t+1} = d_t - 1

Objective
    max over pi of E[ sum_t p_t S_{t+1} ], the expected revenue over the selling window,
    reported with the sell-out probability and the spread of revenue, because unsold
    seats are worth nothing on the day and a promoter cares about both.
"""
import numpy as np

from . import config


def arrival_rate(elapsed: float, a: float = None, b: float = None) -> float:
    """Expected arrivals on a day, as a function of how much of the window has elapsed."""
    a = config.ARRIVAL_A if a is None else a
    b = config.ARRIVAL_B if b is None else b
    return float(a * np.exp(b * elapsed))


def buys(wtp: np.ndarray, price: float, reference: float,
         eta: float = None, gamma: float = None) -> np.ndarray:
    """The brief's purchase rule, applied to a day's arrivals at once."""
    eta = config.ETA if eta is None else eta
    gamma = config.GAMMA if gamma is None else gamma
    threshold = (price + eta * max(price - reference, 0.0)
                 - gamma * max(reference - price, 0.0))
    return wtp >= threshold


def simulate_path(policy, seed: int, capacity: int = None, days: int = None,
                  arrival_a: float = None, arrival_b: float = None,
                  wtp_median: float = None, wtp_log_sd: float = None,
                  theta: float = None, eta: float = None, gamma: float = None,
                  kappa: float = None, scarcity_share: float = None):
    """Run one selling window day by day and return what the metrics need.

    `policy` is called as policy(t, state) and returns the price for the day.  The
    arrivals are drawn from a generator seeded per scenario, so two policies given the
    same seed meet the same buyers in the same order until their own prices change what
    those buyers do, which is as close to common random numbers as a problem with an
    endogenous arrival count allows.
    """
    capacity = config.CAPACITY if capacity is None else capacity
    days = config.SELLING_DAYS if days is None else days
    wtp_median = config.WTP_MEDIAN if wtp_median is None else wtp_median
    wtp_log_sd = config.WTP_LOG_SD if wtp_log_sd is None else wtp_log_sd
    theta = config.THETA if theta is None else theta
    kappa = config.KAPPA if kappa is None else kappa
    scarcity_share = config.SCARCITY_SHARE if scarcity_share is None else scarcity_share

    rng = np.random.default_rng(seed)
    seats = int(capacity)
    reference = float(config.REFERENCE_START)
    revenue = 0.0
    sold_path = np.zeros(days, dtype=int)
    price_path = np.zeros(days)
    seats_path = np.zeros(days + 1, dtype=int)
    seats_path[0] = seats

    for t in range(days):
        state = {"seats": seats, "days_left": days - t, "reference": reference,
                 "capacity": capacity, "day": t,
                 "recent_sales": sold_path[max(0, t - 7):t]}
        price = float(policy(t, state))

        elapsed = t / (days - 1) if days > 1 else 1.0
        rate = arrival_rate(elapsed, arrival_a, arrival_b)
        if seats < scarcity_share * capacity:
            rate *= kappa
        arrivals = int(rng.poisson(rate))

        sold = 0
        if arrivals > 0 and seats > 0:
            wtp = rng.lognormal(np.log(wtp_median), wtp_log_sd, arrivals)
            sold = int(min(seats, int(buys(wtp, price, reference, eta, gamma).sum())))

        revenue += price * sold
        seats -= sold
        reference = theta * reference + (1.0 - theta) * price
        sold_path[t] = sold
        price_path[t] = price
        seats_path[t + 1] = seats

    return {"revenue": revenue,
            "sold": int(capacity - seats),
            "seats_left": seats,
            "sold_out": bool(seats == 0),
            "sold_path": sold_path,
            "price_path": price_path,
            "seats_path": seats_path,
            "average_price": revenue / max(1, capacity - seats)}


def run_policy(make_policy, seeds, **kwargs):
    """Apply one policy to every scenario and collect the metrics for all of them.

    `make_policy` is a factory, so a policy that remembers something gets a fresh copy
    for each scenario, and `seeds` is the shared list of scenario seeds every policy in a
    comparison is given.
    """
    runs = [simulate_path(make_policy(), int(seed), **kwargs) for seed in seeds]
    return {
        "revenue": np.array([r["revenue"] for r in runs]),
        "sold": np.array([r["sold"] for r in runs]),
        "sold_out": np.array([r["sold_out"] for r in runs], dtype=float),
        "average_price": np.array([r["average_price"] for r in runs]),
        "price_paths": np.array([r["price_path"] for r in runs]),
        "seats_paths": np.array([r["seats_path"] for r in runs]),
        "sold_paths": np.array([r["sold_path"] for r in runs]),
    }


def scenario_seeds(seed: int, n: int) -> np.ndarray:
    """One list of scenario seeds, drawn once and reused by every policy."""
    return np.random.default_rng(seed).integers(0, 2**31 - 1, size=n)
