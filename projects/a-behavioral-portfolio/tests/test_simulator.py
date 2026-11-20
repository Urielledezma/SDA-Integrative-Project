"""The properties the report's conclusions rest on.

Asserted qualitatively rather than to five decimals, so a change of library version does
not fail a test whose conclusion still holds.  A change that breaks one of these has
changed a finding, and the report has to be re-read rather than the test rewritten.
"""
import numpy as np
import pytest

from bpd import config as c
from bpd import data, evaluate, policies, questionnaire, simulator


@pytest.fixture(scope="module")
def client():
    return questionnaire.score(questionnaire.default_answers())


@pytest.fixture(scope="module")
def returns():
    return data.monthly_returns(data.load_prices()).to_numpy()


def flat(n_months, n_assets, value=0.0):
    return np.full((n_months, n_assets), value)


# --- the transition ---------------------------------------------------------

def test_wealth_compounds_at_the_portfolio_return_when_nothing_is_traded():
    target = np.array([0.5, 0.5])
    returns = flat(3, 2, 0.10)
    out = simulator.simulate_path(policies.buy_and_hold(target)(), returns, target,
                                  initial=100.0, cost=0.0)
    assert out["terminal"] == pytest.approx(100.0 * 1.10 ** 3)


def test_weights_drift_toward_whatever_grew_faster():
    target = np.array([0.5, 0.5])
    returns = np.array([[0.20, 0.0]])
    out = simulator.simulate_path(policies.buy_and_hold(target)(), returns, target,
                                  initial=100.0, cost=0.0)
    assert out["terminal_weights"][0] == pytest.approx(0.6 / 1.1)
    assert out["terminal_weights"][1] == pytest.approx(0.5 / 1.1)
    assert out["terminal_weights"].sum() == pytest.approx(1.0)


def test_holding_never_trades_so_it_never_pays_a_cost():
    target = np.array([0.5, 0.5])
    returns = np.array([[0.30, -0.10], [0.05, 0.40]])
    out = simulator.simulate_path(policies.buy_and_hold(target)(), returns, target,
                                  initial=100.0, cost=0.01)
    assert out["cost_total"] == 0.0
    assert out["months_traded"] == 0


def test_the_cost_is_charged_on_every_unit_that_changes_hands():
    target = np.array([1.0, 0.0])
    other = np.array([0.0, 1.0])

    def make():
        return lambda t, state: other if t == 0 else state["weights"]

    returns = flat(1, 2, 0.0)
    out = simulator.simulate_path(make(), returns, target, initial=100.0, cost=0.01)
    # one unit sold and one unit bought is a traded total of two, so the charge is 2%.
    assert out["terminal"] == pytest.approx(98.0)
    assert out["turnover_per_year"] == pytest.approx(0.5 * 2.0 / (1 / 12))


def test_the_peak_is_what_the_drawdown_is_measured_against():
    target = np.array([1.0])
    returns = np.array([[0.5], [-0.5]])
    out = simulator.simulate_path(policies.buy_and_hold(target)(), returns, target,
                                  initial=100.0, cost=0.0)
    assert out["terminal"] == pytest.approx(75.0)
    assert out["max_drawdown"] == pytest.approx(0.5)


# --- the policies -----------------------------------------------------------

def test_a_band_restores_the_target_once_the_drift_passes_it():
    target = np.array([0.5, 0.5])
    returns = np.array([[0.40, 0.0], [0.0, 0.0]])
    tight = simulator.simulate_path(policies.band_rebalance(target, 0.01)(), returns,
                                    target, initial=100.0, cost=0.0)
    wide = simulator.simulate_path(policies.band_rebalance(target, 0.40)(), returns,
                                   target, initial=100.0, cost=0.0)
    assert tight["terminal_weights"] == pytest.approx(target)
    assert tight["months_traded"] == 1
    assert wide["months_traded"] == 0


def test_a_wider_band_never_trades_more_often(returns, client):
    target = np.zeros(returns.shape[1])
    target[:5] = 0.2
    paths = data.block_bootstrap(returns, 25, 60, c.EVAL_SEED)
    traded = []
    for band in (0.01, 0.05, 0.20, 0.50):
        out = simulator.run_policy(policies.band_rebalance(target, band), paths, target,
                                   client, 2.0 * c.INITIAL_WEALTH)
        traded.append(out["months_traded"].mean())
    assert traded == sorted(traded, reverse=True)


def test_panic_selling_moves_to_cash_after_the_fall_and_returns_after_the_recovery():
    target = np.array([1.0, 0.0])
    cash_index = 1
    returns = np.array([[-0.25, 0.0], [0.0, 0.0], [0.50, 0.0], [0.0, 0.0]])
    policy = policies.panic_selling(target, cash_index, 0.20, 0.10)()
    out = simulator.simulate_path(policy, returns, target, initial=100.0, cost=0.0)
    # It sells into cash after the 25% fall and so does not take part in the rebound,
    # then the untouched index recovers more than 10% from its low and it goes back in.
    assert out["months_traded"] == 2
    assert out["terminal_weights"][0] == pytest.approx(1.0)


def test_panic_selling_does_nothing_when_the_fall_never_arrives():
    target = np.array([1.0, 0.0])
    returns = flat(12, 2, 0.01)
    out = simulator.simulate_path(policies.panic_selling(target, 1, 0.20, 0.10)(),
                                  returns, target, initial=100.0, cost=0.0)
    assert out["months_traded"] == 0


def test_each_scenario_gets_a_policy_with_its_own_memory(returns, client):
    """A panic rule that stayed parked between scenarios would corrupt every later one."""
    target = np.zeros(returns.shape[1])
    target[:5] = 0.2
    paths = data.block_bootstrap(returns, 30, 60, c.EVAL_SEED)
    first = simulator.run_policy(policies.panic_selling(target, 0, 0.05, 0.05), paths,
                                 target, client, 2.0 * c.INITIAL_WEALTH)
    second = simulator.run_policy(policies.panic_selling(target, 0, 0.05, 0.05), paths,
                                  target, client, 2.0 * c.INITIAL_WEALTH)
    assert first["terminal"] == pytest.approx(second["terminal"])


def test_chasing_tilts_toward_the_winner_and_stays_fully_invested():
    target = np.array([0.5, 0.5])
    history = np.zeros((13, 2))
    history[:, 0] = 0.05
    policy = policies.performance_chasing(target, 1.0, every=12, lookback=12)()
    state = {"weights": target.copy(), "history": history[:12], "wealth": 100.0,
             "peak": 100.0, "index": 100.0, "target": target}
    tilted = policy(12, state)
    assert tilted.sum() == pytest.approx(1.0)
    assert tilted[0] > target[0] > tilted[1]


def test_chasing_does_nothing_before_a_full_window_exists():
    target = np.array([0.5, 0.5])
    policy = policies.performance_chasing(target, 1.0, every=1, lookback=12)()
    state = {"weights": target.copy(), "history": np.zeros((3, 2)), "wealth": 100.0,
             "peak": 100.0, "index": 100.0, "target": target}
    assert policy(3, state) == pytest.approx(target)


# --- the uncertainty model --------------------------------------------------

def test_the_bootstrap_keeps_the_shape_and_the_moments_of_its_sample(returns):
    paths = data.block_bootstrap(returns, 200, 120, c.EVAL_SEED)
    assert paths.shape == (200, 120, returns.shape[1])
    assert np.abs(paths.mean((0, 1)) - returns.mean(0)).max() < 0.004
    assert np.abs(paths.std((0, 1)) - returns.std(0)).max() < 0.004


def test_the_bootstrap_keeps_the_cross_section(returns):
    paths = data.block_bootstrap(returns, 200, 120, c.EVAL_SEED)
    sample = np.corrcoef(returns, rowvar=False)
    simulated = np.corrcoef(paths.reshape(-1, returns.shape[1]), rowvar=False)
    assert np.abs(sample - simulated).max() < 0.10


def test_equalising_means_removes_the_spread_but_not_the_covariance(returns):
    equalised = data.equalise_means(returns)
    assert equalised.mean(0).std() == pytest.approx(0.0, abs=1e-15)
    assert np.abs(np.cov(equalised, rowvar=False) - np.cov(returns, rowvar=False)).max() < 1e-12


def test_one_seed_gives_one_set_of_scenarios_and_a_different_seed_does_not(returns):
    first = data.block_bootstrap(returns, 20, 60, c.EVAL_SEED)
    again = data.block_bootstrap(returns, 20, 60, c.EVAL_SEED)
    other = data.block_bootstrap(returns, 20, 60, c.TUNE_SEED)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_tuning_and_evaluation_seeds_are_not_the_same():
    assert c.TUNE_SEED != c.EVAL_SEED


# --- the evaluation ---------------------------------------------------------

def test_a_paired_difference_refuses_mismatched_samples():
    with pytest.raises(ValueError):
        evaluate.paired_difference(np.zeros(10), np.zeros(9))


def test_a_confidence_interval_brackets_the_mean_and_narrows_with_more_scenarios():
    rng = np.random.default_rng(0)
    small = evaluate.summarise(rng.normal(size=100))
    large = evaluate.summarise(rng.normal(size=10_000))
    assert small["ci_low"] < small["mean"] < small["ci_high"]
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_pairing_is_tighter_than_comparing_two_independent_summaries(returns, client):
    """The point of common random numbers, stated as a test."""
    target = np.zeros(returns.shape[1])
    target[:5] = 0.2
    paths = data.block_bootstrap(returns, 100, 60, c.EVAL_SEED)
    reference = 2.0 * c.INITIAL_WEALTH
    hold = simulator.run_policy(policies.buy_and_hold(target), paths, target, client, reference)
    band = simulator.run_policy(policies.band_rebalance(target, 0.02), paths, target,
                                client, reference)
    paired = evaluate.paired_difference(band["terminal"], hold["terminal"])
    unpaired_width = 2 * evaluate.Z95 * np.hypot(evaluate.summarise(band["terminal"])["se"],
                                                 evaluate.summarise(hold["terminal"])["se"])
    assert (paired["ci_high"] - paired["ci_low"]) < unpaired_width


def test_the_mandate_check_reads_the_quantile_and_not_the_average():
    kept = {"terminal_weights": np.full((100, 3), [0.30, 0.35, 0.35])}
    breached = {"terminal_weights": np.vstack([np.full((90, 3), [0.30, 0.35, 0.35]),
                                               np.full((10, 3), [0.80, 0.10, 0.10])])}
    assert evaluate.mandate_is_kept(kept, 0.35)
    assert not evaluate.mandate_is_kept(breached, 0.35)
