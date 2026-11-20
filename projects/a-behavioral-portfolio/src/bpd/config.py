"""Every parameter and every seed for Project A, in one place.

Values taken from the course are marked with their source, because the report has to
defend each one and a number invented here would have nothing behind it.
"""

# --- The investable universe -------------------------------------------------
# Twenty five exchange traded funds across five asset classes, which satisfies the
# brief's 20 to 30 instruments over at least 4 classes.  All have price history from
# 2016, so the monthly sample is the same length for every one of them.
ASSET_CLASS = {
    "SPY": "US equity", "QQQ": "US equity", "IWM": "US equity",
    "XLK": "US equity", "XLF": "US equity", "XLE": "US equity",
    "XLV": "US equity", "XLP": "US equity", "XLY": "US equity",
    "XLI": "US equity", "XLU": "US equity", "XLB": "US equity",
    "XLRE": "US equity",
    "EFA": "Non-US equity", "EEM": "Non-US equity",
    "TLT": "Fixed income", "IEF": "Fixed income", "LQD": "Fixed income",
    "HYG": "Fixed income", "TIP": "Fixed income", "SHY": "Fixed income",
    "GLD": "Real assets", "DBC": "Real assets", "VNQ": "Real assets",
    "BIL": "Cash",
}
TICKERS = list(ASSET_CLASS)
CLASSES = sorted(set(ASSET_CLASS.values()))

DATA_START = "2016-01-01"
DATA_END = "2026-11-01"

# --- Prospect theory ---------------------------------------------------------
# Tversky and Kahneman (1992), as given in the course deck "003 Prospect Theory",
# page 8 for the value function and page 13 for the weighting function.
ALPHA_TK = 0.88
BETA_TK = 0.88
LAMBDA_TK = 2.25
GAMMA_PLUS = 0.61
GAMMA_MINUS = 0.69

# The same deck, page 24, reports that published loss aversion estimates run from
# 1.4 to 3.0 across populations, stakes and elicitation methods.  An elicited value
# outside it is kept and flagged, since it describes an unusual client rather than an
# impossible one, while the risk aversion of Part 2 is held at the nearer anchor.
LAMBDA_RANGE = (1.4, 3.0)

# OURS, not the course's: the course gives alpha = beta = 0.88 and notes only that
# curvature is below one.  A floor of 0.5 keeps a client who names a very small
# certainty equivalent from producing a value function so flat that the simulation
# cannot tell two outcomes apart, and a ceiling of 1.0 rules out risk seeking over
# gains, which the power form cannot represent together with the published arm.
CURVATURE_RANGE = (0.5, 1.0)

# The mixed lottery used to elicit loss aversion, which is the opening exercise of
# that deck: heads you lose LOSS, tails you win G, and the smallest acceptable G
# identifies lambda through lambda = G**alpha / LOSS**beta (page 11).
ELICIT_LOSS = 1_000.0

# The appendix of that deck, page 31, calls the estimate above biased, because it
# sets both decision weights to one half when in fact they are 0.421 and 0.454, and
# gives the corrected identity.  Applied by default: reporting a number the course
# itself calls biased is not a position we could hold in a viva.
CORRECT_LOSS_AVERSION = True

# --- Portfolio construction --------------------------------------------------
# Mean-variance in the utility form of deck "002 mean variance", page 6:
# maximise w'mu - (gamma / 2) w'Sigma w subject to w'1 = 1.
# The sample mean is shrunk toward its cross-sectional average before it reaches the
# optimiser.  Deck "002 mean variance" page 27 shows that a binding long-only constraint
# is algebraically a covariance shrinkage, and prescribes shrinking on purpose rather
# than leaving it to the constraint, while page 20 reports that errors in means cost ten
# times what errors in covariances cost.  The amount is OURS: the deck says to shrink
# deliberately and names no intensity, so the report shows the whole range from zero
# to one and this is the midpoint of it.
MU_SHRINK = 0.50

# OURS, not the course's and not the brief's.  The brief fixes which constraints the
# model carries and leaves their levels open, so these four are the advisory firm's
# mandate: hold at most ten funds, take no position below 2% because a smaller one
# cannot move the portfolio and still costs a trade, none above 35%, and no more than
# 40% in any one asset class.  The report shows what the position cap does.
K_MAX = 10
MIN_POSITION = 0.02
MAX_POSITION = 0.35
CLASS_CAP = 0.40

# How far beyond the position cap a policy may let the largest holding drift before the
# mandate counts as abandoned, measured at the 95th percentile across scenarios.  OURS,
# and it exists because no policy can hold the cap exactly: the weights move between the
# dates a policy acts, and the optimiser places the portfolio within a point of the cap
# to begin with, so a test for never exceeding it would reject every rebalancing rule,
# including one that trades every month.  What a policy can control is how far past the
# cap it lets the portfolio run, and this is the bound on that overshoot.
DRIFT_TOLERANCE = 0.05

# Risk aversion is not elicited directly.  It is set from loss aversion, so that a
# client who needs a larger gain before accepting a symmetric bet is given a more
# conservative portfolio.  The map is linear through the two anchors below and is a
# modelling choice of ours, stated as such in the report.
GAMMA_AT_LAMBDA_LOW = 2.0
GAMMA_AT_LAMBDA_HIGH = 12.0

# --- Simulation --------------------------------------------------------------
INITIAL_WEALTH = 100_000.0
HORIZON_MONTHS = 120
BLOCK_MIN = 3
BLOCK_MAX = 6
COST_PER_TRADE = 0.0010

# The reference point of the objective is the client's target, not the wealth they start
# with, because with a nominal reference at the initial wealth no scenario in this sample
# ends below it, the loss arm of the value function never activates, and loss aversion
# would have no effect on the ranking.  The report quotes all three multiples and
# headlines the largest, where a shortfall occurs often enough for lambda to matter.
TARGET_MULTIPLES = (2.0, 3.0, 4.0)
HEADLINE_TARGET_MULTIPLE = 4.0

# --- Policy parameters -------------------------------------------------------
# Tuned by policy search on the tuning scenarios.
# The grids run past the point where the policy stops acting, so an optimum that sits
# at an endpoint is recognised as an endpoint rather than mistaken for an interior one.
BAND_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50)
RISK_PENALTY_GRID = (0.25, 0.5, 1.0, 3.0, 6.0, 10.0)
RESOLVE_EVERY_GRID = (6, 12, 24)

# OURS, and descriptive rather than tuned.  No question in the instrument measures how
# far a client lets a portfolio fall before selling or how hard they chase, so these are
# plausible values rather than elicited ones, and the report varies each of them instead
# of resting on the value below.  They are never tuned against the objective, because the
# point of a behavioural policy is to measure what a tendency costs rather than to make
# it perform well.
PANIC_DRAWDOWN = 0.15
PANIC_RECOVERY = 0.10
PANIC_DRAWDOWN_GRID = (0.10, 0.15, 0.20, 0.25)
CHASE_INTENSITY = 0.50
CHASE_EVERY = 12
CHASE_INTENSITY_GRID = (0.25, 0.50, 1.00)

# --- Seeds and sample sizes --------------------------------------------------
# The brief requires separate seeds for tuning and for the final evaluation, and the
# course deck "005 SDA" page 18 requires that every policy in a comparison see the
# same scenarios, so the evaluation draws one set of paths and reuses it.
TUNE_SEED = 20261119
EVAL_SEED = 20261124
# Tuning needs less precision than reporting, because it only has to rank parameter
# settings against each other, so it uses fewer scenarios while the final evaluation
# keeps the 500 the brief requires.
N_TUNE = 100
N_EVAL = 500
