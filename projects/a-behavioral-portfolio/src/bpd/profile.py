"""Part 1: the client profile.

Ten questions estimate the three parameters of the prospect-theory value function of
deck "003 Prospect Theory" page 8,

    v(x) = x**alpha            for x >= 0
    v(x) = -lambda (-x)**beta  for x < 0

and five scenario questions set the behavioural flags reported alongside them.

The three parameters are identified in the order beta, alpha, lambda, because the loss
arm of a loss-only lottery cancels lambda, which leaves curvature alone, and the mixed
lottery then identifies lambda given both curvatures.  Probabilities enter raw rather
than weighted, which is what the deck does on page 11 when it writes the indifference
condition for the opening exercise as 0.5 G**0.88 = 0.5 lambda (1000)**0.88.
"""
import numpy as np

from . import config


def curvature_from_ce(certainty_equivalent: float, prize: float) -> float:
    """Curvature implied by the certainty equivalent of a fifty-fifty lottery.

    For a gain-only lottery paying `prize` or nothing, v(CE) = 0.5 v(prize) gives
    CE**alpha = 0.5 prize**alpha, so alpha = ln(0.5) / ln(CE / prize).  The loss-only
    lottery gives the same expression for beta, with both quantities as magnitudes,
    because lambda multiplies the two sides equally and cancels.
    """
    ratio = certainty_equivalent / prize
    if not 0.0 < ratio < 1.0:
        return float("nan")
    return float(np.log(0.5) / np.log(ratio))


def loss_aversion_from_g(accept_gain: float, alpha: float, beta: float,
                         loss: float = None, corrected: bool = None) -> float:
    """Loss aversion implied by the smallest acceptable gain on a fifty-fifty bet.

    Indifference on win `accept_gain` against lose `loss` gives
    0.5 G**alpha = 0.5 lambda L**beta, so lambda = G**alpha / L**beta, and with
    alpha = beta = 0.88 and L = 1000 that is the deck's lambda = (G / 1000)**0.88,
    whose reference values on page 11 the tests reproduce.

    The appendix of the same deck, page 31, says that estimate is biased, because
    setting both decision weights to one half is wrong when the two are distorted
    unequally, at w+(0.5) = 0.421 against w-(0.5) = 0.454.  Its corrected identity is

        lambda = (w+(0.5) / w-(0.5)) * G**alpha / L**beta

    which is about 7% lower, and the appendix's worked case of G = 2,500 giving 2.08
    rather than 2.24 is asserted in the tests.  The correction is applied by default,
    since reporting the number the course itself calls biased would be hard to defend.
    """
    loss = config.ELICIT_LOSS if loss is None else loss
    corrected = config.CORRECT_LOSS_AVERSION if corrected is None else corrected
    if accept_gain <= 0 or loss <= 0:
        return float("nan")
    naive = accept_gain ** alpha / loss ** beta
    if not corrected:
        return float(naive)
    return float(naive * weight(0.5, config.GAMMA_PLUS) / weight(0.5, config.GAMMA_MINUS))


def value(x, alpha: float, beta: float, lam: float):
    """The prospect-theory value of a gain or loss measured from the reference point."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, np.abs(x) ** alpha, -lam * np.abs(x) ** beta)


def weight(p, gamma: float):
    """The Tversky-Kahneman probability weighting function, deck page 13.

    Reported in the profile as a diagnostic, since the estimates above use raw
    probabilities and the dashboard shows how far a client's stated probabilities would
    be distorted.
    """
    p = np.asarray(p, dtype=float)
    return p ** gamma / (p ** gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)


def risk_aversion(lam: float) -> float:
    """Mean-variance risk aversion gamma implied by loss aversion.

    A modelling choice of ours, not a result from the course: gamma runs linearly
    between the two anchors in config as lambda runs across its published range, so a
    client who demands a larger gain before accepting a symmetric bet is handed a more
    conservative portfolio in Part 2.
    """
    lo, hi = config.LAMBDA_RANGE
    t = (float(np.clip(lam, lo, hi)) - lo) / (hi - lo)
    return config.GAMMA_AT_LAMBDA_LOW + t * (config.GAMMA_AT_LAMBDA_HIGH - config.GAMMA_AT_LAMBDA_LOW)
