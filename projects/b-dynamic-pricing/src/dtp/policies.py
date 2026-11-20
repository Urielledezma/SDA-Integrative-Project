"""Step 5 for Project B: the policies, and the class each one belongs to.

The baseline is the best single fixed price found by simulation, which is what the brief
asks for.  The schedule is a PFA, since it is a closed-form function of days left and
seats remaining with no optimisation at decision time.  The lookahead is a DLA, since it
writes down an explicit model of the remaining days, optimises over it, and implements
only today's price, which is the definition on the course's own slide and the pattern its
rolling-horizon picture draws.

Every factory returns a fresh policy, so nothing carries from one scenario to the next.
"""
import numpy as np
from scipy.stats import norm

from . import config

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except ImportError:                                   # pragma: no cover
    HAVE_GUROBI = False


def snap(price: float, grid=None) -> float:
    """Round to the nearest price the promoter is allowed to charge."""
    grid = config.PRICE_GRID if grid is None else grid
    return float(min(grid, key=lambda p: abs(p - price)))


def fixed_price(price: float):
    """Baseline. One price for the whole window, chosen by simulation."""
    price = snap(price)

    def make():
        def policy(t, state):
            return price
        return policy
    return make


def price_schedule(start: float, slope: float, scarcity: float):
    """PFA. A declining schedule corrected by how fast the seats are going.

    Tunable parameters: `start`, the opening price; `slope`, how much of the way down to
    the floor the schedule travels by the day of the event; and `scarcity`, how hard the
    price reacts to selling ahead of or behind a straight line.

    The correction compares seats remaining against a straight sell-down, so a promoter
    sitting on more inventory than that line cuts the price and one selling ahead of it
    raises the price, which is the behaviour the brief asks a schedule to show without
    any optimisation at decision time.
    """
    def make():
        def policy(t, state):
            days = state["days_left"] + t
            elapsed = t / (days - 1) if days > 1 else 1.0
            base = start - slope * (start - config.PRICE_MIN) * elapsed
            on_track = state["capacity"] * (1.0 - elapsed)
            pressure = (state["seats"] - on_track) / state["capacity"]
            price = base - scarcity * pressure * (config.PRICE_MAX - config.PRICE_MIN)
            return snap(float(np.clip(price, config.PRICE_MIN, config.PRICE_MAX)))
        return policy
    return make


def expected_sales(price: float, reference: float, elapsed: float,
                   arrival_a: float = None, arrival_b: float = None,
                   wtp_median: float = None, wtp_log_sd: float = None) -> float:
    """Expected tickets sold in one day at one price, under the demand model.

    The arrivals are Poisson so their expectation is the rate, and each arrival buys when
    its willingness to pay clears the threshold, which for a lognormal willingness to pay
    is one minus the normal distribution function at the logged threshold.
    """
    from .simulator import arrival_rate
    wtp_median = config.WTP_MEDIAN if wtp_median is None else wtp_median
    wtp_log_sd = config.WTP_LOG_SD if wtp_log_sd is None else wtp_log_sd
    threshold = (price + config.ETA * max(price - reference, 0.0)
                 - config.GAMMA * max(reference - price, 0.0))
    if threshold <= 0:
        share = 1.0
    else:
        share = float(1.0 - norm.cdf((np.log(threshold) - np.log(wtp_median)) / wtp_log_sd))
    return arrival_rate(elapsed, arrival_a, arrival_b) * share


def lookahead(horizon: int = 10, arrival_a: float = None, arrival_b: float = None):
    """DLA. Each day, solve a deterministic model of the days ahead and use today's price.

    Tunable parameter: `horizon`, the number of days the model looks over, which the
    course lists among the things a direct lookahead has to tune.

    The lookahead model is approximate in three ways, and each is a tilde variable in the
    course's notation rather than an oversight.  The horizon is truncated.  Demand is
    replaced by its expectation, so the Poisson draw and the spread of willingness to pay
    both disappear.  And the reference price is held at today's value across the horizon,
    because letting it follow the prices being chosen would make the sales term depend on
    the decisions through a product and the model would stop being a mixed-integer linear
    programme.

    The programme itself chooses one price per day from the grid, with the total expected
    sales over the horizon capped by the seats that remain.
    """
    def make():
        def policy(t, state):
            seats = state["seats"]
            if seats <= 0:
                return config.PRICE_MAX
            days = state["days_left"]
            steps = int(min(horizon, days))
            total_days = days + t
            grid = config.PRICE_GRID

            demand = np.array([[expected_sales(p, state["reference"],
                                               min(1.0, (t + k) / max(1, total_days - 1)),
                                               arrival_a, arrival_b)
                                for p in grid] for k in range(steps)])

            if not HAVE_GUROBI:                       # pragma: no cover
                revenue = demand * np.array(grid)[None, :]
                return float(grid[int(np.argmax(revenue[0]))])

            model = gp.Model()
            model.Params.OutputFlag = 0
            y = model.addVars(steps, len(grid), vtype=GRB.BINARY, name="y")
            for k in range(steps):
                model.addConstr(gp.quicksum(y[k, i] for i in range(len(grid))) == 1)
            model.addConstr(gp.quicksum(demand[k, i] * y[k, i]
                                        for k in range(steps)
                                        for i in range(len(grid))) <= seats)
            model.setObjective(gp.quicksum(grid[i] * demand[k, i] * y[k, i]
                                           for k in range(steps)
                                           for i in range(len(grid))), GRB.MAXIMIZE)
            model.optimize()
            if model.SolCount == 0:                   # pragma: no cover
                return float(grid[0])
            chosen = max(range(len(grid)), key=lambda i: y[0, i].X)
            return float(grid[chosen])
        return policy
    return make
