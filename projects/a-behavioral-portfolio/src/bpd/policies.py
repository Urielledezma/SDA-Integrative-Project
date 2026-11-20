"""Step 5 of the six-step process: the policies, and the class each one belongs to.

Deck "007 Policy Classes" page 6 places rebalancing bands and stop-loss rules in the PFA
column and a myopic optimisation with tuned penalties in the CFA column, and page 11
calls that second object "a MILP with knobs".  Page 22 adds that the rolling version of
it is a CFA and DLA hybrid which collapses to the single-period CFA at horizon zero,
which is the version used here.

The two behavioural policies are PFAs in structure, since each is a threshold rule read
off the state, and deck "005 SDA" page 27 makes exactly that point about the disposition
effect: a behavioural regularity "maps a situation to a decision, and can therefore be
written down and evaluated like any other rule".  What separates them from the PFA band
is purpose rather than form, because their parameters describe the client instead of
being chosen to perform well, so they are never tuned against the objective.

Every factory returns a fresh policy, because a policy that remembers whether the client
is sitting in cash must not carry that memory from one scenario into the next.
"""
import numpy as np

from . import config, portfolio


def buy_and_hold(target: np.ndarray):
    """Baseline. Hold the Part 2 portfolio and never trade, so the weights drift.

    Class: none, it is the do-nothing rule the brief asks for as the simple baseline, and
    it is the "Hold, never trade" row of deck "005 SDA" page 7.
    """
    target = np.asarray(target, dtype=float)

    def make():
        def policy(t, state):
            return state["weights"]
        return policy
    return make


def band_rebalance(target: np.ndarray, band: float):
    """PFA. Restore the target whenever any weight has drifted more than `band`.

    Tunable parameter: `band`, the half-width in weight fractions, which is the theta of
    deck "007 Policy Classes" page 7 and the "Band of 2 points" row of deck "005 SDA"
    page 7.  Trading restores every weight at once rather than only the offending one,
    which is what that row does.
    """
    target = np.asarray(target, dtype=float)

    def make():
        def policy(t, state):
            if np.abs(state["weights"] - target).max() > band:
                return target
            return state["weights"]
        return policy
    return make


def periodic_resolve(target: np.ndarray, mu, sigma, classes, risk_penalty: float,
                     every: int = 12, cost: float = None):
    """CFA. Re-solve the Part 2 model every `every` months with the trading cost inside it.

    Tunable parameters: `risk_penalty`, the theta_1 of deck "007 Policy Classes" page 11,
    and `every`, the number of months between solves.  The penalty is tuned rather than
    taken from the client's profile, because page 11 warns that a CFA is myopic by
    construction and its knobs "are the only channel through which the future enters, so
    they must be tuned, not guessed".

    The mean and covariance are the ones estimated at time zero from real history rather
    than re-estimated from the simulated path, so no decision uses information that was
    not available when it was made.  The decision is still path dependent, because the
    model is charged for moving away from the weights it currently holds and therefore
    returns a partial rebalance whose size depends on how far the portfolio has drifted.
    """
    target = np.asarray(target, dtype=float)
    cost = config.COST_PER_TRADE if cost is None else cost

    def make():
        def policy(t, state):
            if t == 0 or t % every != 0:
                return state["weights"]
            weights, _ = portfolio.solve(mu, sigma, classes, risk_penalty,
                                         w_prev=state["weights"], cost=cost)
            return weights
        return policy
    return make


def panic_selling(target: np.ndarray, cash_index: int, drawdown: float, recovery: float):
    """Behavioural. Move everything to cash after a drawdown, return after a recovery.

    Parameters describe the client rather than being tuned: `drawdown` is the fall from
    peak wealth that triggers the exit and `recovery` is the rise the market has to make
    from its low before the client comes back.

    The exit is read off the client's own wealth against its peak, which is the drawdown
    state the brief asks for.  The return has to be read off the market instead, because
    a client sitting in cash has no drawdown of their own left to recover from, so the
    rule watches the value the untouched target portfolio would have reached, which the
    simulator carries as the informational state I_t.
    """
    target = np.asarray(target, dtype=float)
    cash = np.zeros_like(target)
    cash[cash_index] = 1.0

    def make():
        memory = {"parked": False, "trough": np.inf}

        def policy(t, state):
            if memory["parked"]:
                memory["trough"] = min(memory["trough"], state["index"])
                if state["index"] >= memory["trough"] * (1.0 + recovery):
                    memory["parked"] = False
                    memory["trough"] = np.inf
                    return target
                return state["weights"]
            if state["wealth"] <= state["peak"] * (1.0 - drawdown):
                memory["parked"] = True
                memory["trough"] = state["index"]
                return cash
            return state["weights"]
        return policy
    return make


def performance_chasing(target: np.ndarray, intensity: float, every: int = 12,
                        lookback: int = 12):
    """Behavioural. Tilt the portfolio toward whatever has been winning.

    Parameters describe the client: `intensity` is how far they lean toward the recent
    winners, `every` is how often they act and `lookback` is the window they look at.

    The tilt is multiplicative on the target weights and is renormalised, so the client
    stays fully invested in the assets their plan selected and only changes the mix, and
    the tilt is skipped until a full window of returns exists because before that there
    is nothing for the client to have looked at.
    """
    target = np.asarray(target, dtype=float)
    held = target > 1e-9

    def make():
        def policy(t, state):
            history = state["history"]
            if t == 0 or t % every != 0 or len(history) < lookback:
                return state["weights"]
            cumulative = np.prod(1.0 + history[-lookback:], axis=0) - 1.0
            scores = cumulative[held]
            spread = scores.std()
            if spread < 1e-12:
                return state["weights"]
            z = (scores - scores.mean()) / spread
            tilted = np.zeros_like(target)
            tilted[held] = np.clip(target[held] * (1.0 + intensity * z), 0.0, None)
            if tilted.sum() <= 0.0:
                return state["weights"]
            return tilted / tilted.sum()
        return policy
    return make
