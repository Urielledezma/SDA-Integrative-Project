"""Every parameter and every seed for Project B, in one place.

The setting and the demand model are given in the brief, which specifies them as
equations rather than as code, so the values below that carry a source are transcribed
from it and the rest are marked as ours.
"""

# --- The setting, given by the brief -----------------------------------------
CAPACITY = 2_000
SELLING_DAYS = 60
PRICE_MIN = 40.0
PRICE_MAX = 200.0
PRICE_STEP = 10.0
PRICE_GRID = [PRICE_MIN + PRICE_STEP * i
              for i in range(int((PRICE_MAX - PRICE_MIN) / PRICE_STEP) + 1)]

# --- The demand model, given by the brief ------------------------------------
# Arrivals are Poisson with a rate that rises as the event approaches.  The brief does
# not fix the shape, and requires it to be estimated from the Kaggle flight-prices data,
# so the form below is ours: with u the fraction of the selling window already elapsed,
#
#     lambda(u) = ARRIVAL_A * exp(ARRIVAL_B * u)
#
# which has two parameters, is monotone in u whenever ARRIVAL_B is positive, and is
# linear in its parameters once logged, so it can be fitted by least squares on the
# observed sales rate per day to departure.  The defaults here are replaced by
# calibrate.py once the data has been fitted.
ARRIVAL_A = 20.0
ARRIVAL_B = 1.5

# Willingness to pay is lognormal.  Stated through its median and the log standard
# deviation, because a median in pesos is a quantity a promoter can argue with while a
# mu on the log scale is not.
WTP_MEDIAN = 95.0
WTP_LOG_SD = 0.55

# The reference price updates as r_{t+1} = THETA * r_t + (1 - THETA) * p_t.
THETA = 0.80

# A price above the reference is penalised by ETA and one below is rewarded by GAMMA, so
# a buyer takes a ticket when
#     WTP >= p_t + ETA * max(p_t - r_t, 0) - GAMMA * max(r_t - p_t, 0).
ETA = 0.60
GAMMA = 0.25

# Below this share of seats the arrival rate is multiplied by KAPPA.
SCARCITY_SHARE = 0.10
KAPPA = 1.5

# The reference price the market starts with.  Ours: the midpoint of the grid, since the
# brief gives the updating rule without an initial condition and a buyer who has seen no
# price yet has nothing else to anchor on.
REFERENCE_START = 0.5 * (PRICE_MIN + PRICE_MAX)

# --- Policy parameters -------------------------------------------------------
# Tuned by policy search on the tuning scenarios.
SCHEDULE_START_GRID = (120.0, 140.0, 160.0, 180.0)
SCHEDULE_SLOPE_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
SCHEDULE_SCARCITY_GRID = (0.0, 0.2, 0.4)
LOOKAHEAD_HORIZON_GRID = (5, 10, 20)

# --- Seeds and sample sizes --------------------------------------------------
TUNE_SEED = 20261120
EVAL_SEED = 20261124
N_TUNE = 200
N_EVAL = 500
