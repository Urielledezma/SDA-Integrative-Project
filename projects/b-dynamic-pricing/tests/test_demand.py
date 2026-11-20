"""The demand model must do what the brief says it does.

Every expectation below is a sentence in the brief, so a change that breaks one of these
has changed the model into something the assignment did not describe.
"""
import numpy as np
import pytest

from dtp import config as c
from dtp import policies, simulator


# --- the purchase rule -------------------------------------------------------

def test_a_buyer_takes_the_ticket_exactly_when_it_clears_the_threshold():
    wtp = np.array([99.0, 100.0, 101.0])
    taken = simulator.buys(wtp, price=100.0, reference=100.0, eta=0.0, gamma=0.0)
    assert list(taken) == [False, True, True]


def test_pricing_above_the_reference_raises_the_bar_by_eta():
    wtp = np.array([105.0])
    assert simulator.buys(wtp, 100.0, reference=90.0, eta=0.0, gamma=0.0)[0]
    # Ten above the reference, penalised at 0.6, needs 106 rather than 100.
    assert not simulator.buys(wtp, 100.0, reference=90.0, eta=0.6, gamma=0.0)[0]


def test_pricing_below_the_reference_lowers_the_bar_by_gamma():
    wtp = np.array([97.0])
    assert not simulator.buys(wtp, 100.0, reference=120.0, eta=0.6, gamma=0.0)[0]
    assert simulator.buys(wtp, 100.0, reference=120.0, eta=0.6, gamma=0.25)[0]


# --- the arrival curve -------------------------------------------------------

def test_arrivals_rise_toward_the_event():
    rates = [simulator.arrival_rate(u) for u in np.linspace(0.0, 1.0, 11)]
    assert rates == sorted(rates)
    assert rates[-1] > rates[0]


def test_the_arrival_curve_is_the_two_parameter_exponential():
    assert simulator.arrival_rate(0.0, a=7.0, b=2.0) == pytest.approx(7.0)
    assert simulator.arrival_rate(1.0, a=7.0, b=2.0) == pytest.approx(7.0 * np.exp(2.0))


# --- the transition ----------------------------------------------------------

def test_nothing_sells_when_nobody_arrives():
    out = simulator.simulate_path(policies.fixed_price(200.0)(), seed=1, capacity=10,
                                  days=3, arrival_a=0.0, arrival_b=0.0)
    assert out["sold"] == 0
    assert out["revenue"] == 0.0


def test_seats_never_go_negative_and_revenue_matches_what_was_sold():
    out = simulator.simulate_path(policies.fixed_price(40.0)(), seed=3, capacity=50,
                                  days=20)
    assert out["seats_left"] >= 0
    assert out["sold"] == 50 - out["seats_left"]
    assert out["revenue"] == pytest.approx(40.0 * out["sold"])
    assert out["seats_path"][0] == 50
    assert (np.diff(out["seats_path"]) <= 0).all()


def test_a_sold_out_window_stops_selling():
    out = simulator.simulate_path(policies.fixed_price(40.0)(), seed=5, capacity=20,
                                  days=60)
    assert out["sold_out"]
    assert out["sold"] == 20


def test_scarcity_cannot_bite_while_the_seats_are_plentiful():
    calm = simulator.simulate_path(policies.fixed_price(200.0)(), seed=7,
                                   capacity=5_000, days=30, kappa=1.0)
    eager = simulator.simulate_path(policies.fixed_price(200.0)(), seed=7,
                                    capacity=5_000, days=30, kappa=3.0)
    assert calm["sold"] == eager["sold"]


# --- common random numbers ---------------------------------------------------

def test_one_seed_gives_one_scenario():
    first = simulator.simulate_path(policies.fixed_price(100.0)(), seed=11)
    again = simulator.simulate_path(policies.fixed_price(100.0)(), seed=11)
    other = simulator.simulate_path(policies.fixed_price(100.0)(), seed=12)
    assert first["revenue"] == again["revenue"]
    assert first["revenue"] != other["revenue"]


def test_the_scenario_seeds_are_shared_and_the_two_streams_differ():
    tune = simulator.scenario_seeds(c.TUNE_SEED, 50)
    again = simulator.scenario_seeds(c.TUNE_SEED, 50)
    report = simulator.scenario_seeds(c.EVAL_SEED, 50)
    assert np.array_equal(tune, again)
    assert not np.array_equal(tune, report)
    assert c.TUNE_SEED != c.EVAL_SEED


# --- the policies ------------------------------------------------------------

def test_every_price_a_policy_returns_is_on_the_grid():
    seeds = simulator.scenario_seeds(c.EVAL_SEED, 5)
    for make in (policies.fixed_price(97.0), policies.price_schedule(140.0, 0.5, 0.3)):
        out = simulator.run_policy(make, seeds)
        assert set(np.unique(out["price_paths"])) <= set(c.PRICE_GRID)


def test_a_fixed_price_never_moves():
    out = simulator.simulate_path(policies.fixed_price(120.0)(), seed=2)
    assert len(np.unique(out["price_path"])) == 1


def test_the_schedule_falls_over_the_window_when_it_is_told_to():
    out = simulator.simulate_path(policies.price_schedule(200.0, 1.0, 0.0)(), seed=2,
                                  capacity=10_000)
    assert out["price_path"][0] > out["price_path"][-1]


def test_the_schedule_charges_more_when_the_seats_are_going_faster():
    gentle = policies.price_schedule(140.0, 0.5, 0.0)()
    reactive = policies.price_schedule(140.0, 0.5, 0.4)()
    ahead = {"seats": 200, "days_left": 30, "capacity": 1_000, "day": 30,
             "reference": 100.0, "recent_sales": np.array([])}
    behind = dict(ahead, seats=900)
    assert reactive(30, ahead) > reactive(30, behind)
    assert gentle(30, ahead) == gentle(30, behind)


def test_expected_sales_fall_as_the_price_rises():
    sales = [policies.expected_sales(p, reference=100.0, elapsed=0.5)
             for p in c.PRICE_GRID]
    assert sales == sorted(sales, reverse=True)


def test_expected_sales_rise_toward_the_event():
    early = policies.expected_sales(100.0, reference=100.0, elapsed=0.0)
    late = policies.expected_sales(100.0, reference=100.0, elapsed=1.0)
    assert late > early


def test_the_lookahead_charges_at_least_as_much_when_fewer_seats_remain():
    policy = policies.lookahead(5)()
    plenty = {"seats": 1_500, "days_left": 30, "capacity": 2_000, "day": 30,
              "reference": 120.0, "recent_sales": np.array([])}
    scarce = dict(plenty, seats=40)
    assert policy(30, scarce) >= policy(30, plenty)


def test_the_lookahead_stops_discounting_once_the_seats_are_gone():
    policy = policies.lookahead(5)()
    empty = {"seats": 0, "days_left": 10, "capacity": 2_000, "day": 50,
             "reference": 100.0, "recent_sales": np.array([])}
    assert policy(50, empty) == c.PRICE_MAX


def test_each_scenario_gets_a_policy_with_its_own_memory():
    seeds = simulator.scenario_seeds(c.EVAL_SEED, 10)
    first = simulator.run_policy(policies.price_schedule(140.0, 0.5, 0.3), seeds)
    second = simulator.run_policy(policies.price_schedule(140.0, 0.5, 0.3), seeds)
    assert np.array_equal(first["revenue"], second["revenue"])
