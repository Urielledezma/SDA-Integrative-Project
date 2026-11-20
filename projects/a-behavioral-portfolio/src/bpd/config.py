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
# 1.4 to 3.0 across populations, stakes and elicitation methods, so an answer that
# implies a value outside that range is reported as out of range rather than used.
LAMBDA_RANGE = (1.4, 3.0)
CURVATURE_RANGE = (0.5, 1.0)

# The mixed lottery used to elicit loss aversion, which is the opening exercise of
# that deck: heads you lose LOSS, tails you win G, and the smallest acceptable G
# identifies lambda through lambda = G**alpha / LOSS**beta (page 11).
ELICIT_LOSS = 1_000.0

# --- Portfolio construction --------------------------------------------------
# Mean-variance in the utility form of deck "002 mean variance", page 6:
# maximise w'mu - (gamma / 2) w'Sigma w subject to w'1 = 1.
K_MAX = 10
MIN_POSITION = 0.02
MAX_POSITION = 0.35
CLASS_CAP = 0.40

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

# --- Seeds and sample sizes --------------------------------------------------
# The brief requires separate seeds for tuning and for the final evaluation, and the
# course deck "005 SDA" page 18 requires that every policy in a comparison see the
# same scenarios, so the evaluation draws one set of paths and reuses it.
TUNE_SEED = 20261119
EVAL_SEED = 20261124
N_TUNE = 200
N_EVAL = 500
