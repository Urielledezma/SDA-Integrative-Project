"""The ten-question instrument, and the profile it produces.

Questions 1 to 4 are preference elicitation and feed the three value-function
parameters.  Question 5 fixes the reference point, which deck "003 Prospect Theory"
page 24 calls an input rather than an output, so it has to be asked rather than
estimated.  Questions 6 to 10 are the scenario questions that set the behavioural flags,
and each one names the Part 3 policy it switches on.
"""
from dataclasses import dataclass, field

import numpy as np

from . import config, profile

GAIN_PRIZE = 10_000.0
GAIN_PRIZE_SMALL = 2_000.0
LOSS_PRIZE = 10_000.0

LADDER = [1_200.0, 1_500.0, 1_800.0, 2_200.0, 2_600.0, 3_000.0, 4_000.0]

QUESTIONS = [
    {"id": "q1_ladder", "kind": "ladder",
     "text": "A fair coin is flipped. Heads you lose $1,000. Tails you win G. "
             "What is the smallest G at which you would take the bet?",
     "options": LADDER, "measures": "loss aversion"},
    {"id": "q2_gain_ce", "kind": "amount",
     "text": "A fair coin is flipped. Heads you win $10,000, tails you win nothing. "
             "What certain amount would you accept instead of the bet?",
     "bounds": (100.0, 9_900.0), "default": 4_000.0, "measures": "curvature over gains"},
    {"id": "q3_loss_ce", "kind": "amount",
     "text": "A fair coin is flipped. Heads you lose $10,000, tails you lose nothing. "
             "What certain loss would you accept instead of the bet?",
     "bounds": (100.0, 9_900.0), "default": 4_000.0, "measures": "curvature over losses"},
    {"id": "q4_gain_ce_small", "kind": "amount",
     "text": "The same bet at a smaller stake. Heads you win $2,000, tails nothing. "
             "What certain amount would you accept instead?",
     "bounds": (20.0, 1_980.0), "default": 800.0, "measures": "curvature, consistency check"},
    {"id": "q5_reference", "kind": "choice",
     "text": "Ten years from now, which number will you compare your portfolio against?",
     "options": ["The amount I started with",
                 "A target I have in mind",
                 "The highest value it ever reached"],
     "measures": "reference point"},
    {"id": "q6_drawdown", "kind": "choice",
     "text": "Your portfolio has fallen 20% in three months. What do you do?",
     "options": ["Sell and wait in cash until it recovers",
                 "Sell part of it",
                 "Nothing, I keep the plan",
                 "Buy more at the lower prices"],
     "measures": "panic selling"},
    {"id": "q7_chasing", "kind": "choice",
     "text": "One holding has been the best performer for a year while another has lagged. "
             "What do you do?",
     "options": ["Move money into the one that has been winning",
                 "Nothing, I keep the plan",
                 "Move money into the one that has lagged"],
     "measures": "performance chasing"},
    {"id": "q8_disposition", "kind": "choice",
     "text": "You need to raise cash and must sell one of two holdings. One is above what "
             "you paid for it and one is below. Which do you sell?",
     "options": ["The one that is above what I paid",
                 "Whichever the plan says",
                 "The one that is below what I paid"],
     "measures": "disposition effect"},
    {"id": "q9_monitoring", "kind": "choice",
     "text": "How often would you look at the value of this portfolio?",
     "options": ["Daily", "Monthly", "Once or twice a year"],
     "measures": "narrow framing"},
    {"id": "q10_anchor", "kind": "choice",
     "text": "A holding is 30% below what you paid and your plan no longer calls for it. "
             "What do you do?",
     "options": ["Hold it until it gets back to what I paid",
                 "Sell it and follow the plan"],
     "measures": "anchoring on the purchase price"},
]


@dataclass
class ClientProfile:
    alpha: float
    beta: float
    lam: float
    gamma: float
    reference: str
    flags: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def panic_selling(self) -> bool:
        return self.flags.get("panic_selling", False)

    @property
    def performance_chasing(self) -> bool:
        return self.flags.get("performance_chasing", False)


def _bracket_midpoint(accept_gain: float) -> float:
    """The indifference gain implied by the first rung the client accepts.

    A ladder brackets the switch point rather than locating it, so the midpoint of the
    bracket between the last rejected rung and the accepted one is used, and the lowest
    rung is taken at face value because nothing below it was offered.
    """
    i = LADDER.index(accept_gain)
    return accept_gain if i == 0 else 0.5 * (LADDER[i - 1] + accept_gain)


def score(answers: dict) -> ClientProfile:
    """Turn the ten answers into the parameters and flags Parts 2 and 3 consume."""
    notes = []
    lo_c, hi_c = config.CURVATURE_RANGE

    alpha_big = profile.curvature_from_ce(answers["q2_gain_ce"], GAIN_PRIZE)
    alpha_small = profile.curvature_from_ce(answers["q4_gain_ce_small"], GAIN_PRIZE_SMALL)
    alpha_raw = float(np.nanmean([alpha_big, alpha_small]))
    beta_raw = profile.curvature_from_ce(answers["q3_loss_ce"], LOSS_PRIZE)

    if np.isfinite(alpha_big) and np.isfinite(alpha_small) and abs(alpha_big - alpha_small) > 0.15:
        notes.append("The two gain questions imply curvatures that differ by more than 0.15, "
                     "so the average is reported and the estimate is less reliable than usual.")

    alpha = float(np.clip(alpha_raw, lo_c, hi_c))
    beta = float(np.clip(beta_raw, lo_c, hi_c))
    for name, raw, clipped in (("gains", alpha_raw, alpha), ("losses", beta_raw, beta)):
        if np.isfinite(raw) and abs(raw - clipped) > 1e-9:
            notes.append(f"Curvature over {name} came out at {raw:.2f} and was held at "
                         f"{clipped:.2f}, the edge of the range the course reports.")

    gain = _bracket_midpoint(answers["q1_ladder"])
    lam = profile.loss_aversion_from_g(gain, alpha, beta)
    lo_l, hi_l = config.LAMBDA_RANGE
    if not lo_l <= lam <= hi_l:
        notes.append(f"Loss aversion of {lam:.2f} falls outside the 1.4 to 3.0 range the course "
                     f"reports for published estimates, so it is an unusual client rather than an "
                     f"impossible one, and the portfolio's risk aversion is held at the nearer "
                     f"anchor.")

    flags = {
        "panic_selling": answers["q6_drawdown"] in (
            "Sell and wait in cash until it recovers", "Sell part of it"),
        "performance_chasing": answers["q7_chasing"] == "Move money into the one that has been winning",
        "disposition": answers["q8_disposition"] == "The one that is above what I paid",
        "narrow_framing": answers["q9_monitoring"] == "Daily",
        "anchoring": answers["q10_anchor"] == "Hold it until it gets back to what I paid",
    }
    return ClientProfile(alpha=alpha, beta=beta, lam=lam, gamma=profile.risk_aversion(lam),
                         reference=answers["q5_reference"], flags=flags, notes=notes)


def default_answers() -> dict:
    """The answers of the client the report quotes throughout.

    The three certainty equivalents are the exact inversions of the curvature formula at
    the published alpha = beta = 0.88, so this client's curvatures come back at 0.88 to
    three decimals.  Loss aversion lands at 2.16 rather than the published 2.25, because
    the ladder brackets the switch point instead of locating it and no rung sits at the
    $2,513 that 2.25 would require.  Every behavioural flag is set, which makes this the
    client whose tendencies cost the most and therefore the one worth showing a dashboard.
    """
    ce_gain = GAIN_PRIZE * 0.5 ** (1.0 / config.ALPHA_TK)
    ce_gain_small = GAIN_PRIZE_SMALL * 0.5 ** (1.0 / config.ALPHA_TK)
    ce_loss = LOSS_PRIZE * 0.5 ** (1.0 / config.BETA_TK)
    return {
        "q1_ladder": 2_600.0,
        "q2_gain_ce": round(ce_gain, 2),
        "q3_loss_ce": round(ce_loss, 2),
        "q4_gain_ce_small": round(ce_gain_small, 2),
        "q5_reference": "The amount I started with",
        "q6_drawdown": "Sell and wait in cash until it recovers",
        "q7_chasing": "Move money into the one that has been winning",
        "q8_disposition": "The one that is above what I paid",
        "q9_monitoring": "Daily",
        "q10_anchor": "Hold it until it gets back to what I paid",
    }
