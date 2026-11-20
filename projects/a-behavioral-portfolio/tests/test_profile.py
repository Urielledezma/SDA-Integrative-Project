"""The client profile must reproduce the course deck's own published numbers.

Every expected value below is printed on a slide of "003 Prospect Theory", so a change
that breaks one of these has changed the elicitation into something the course did not
teach, and the report would no longer be defensible.
"""
import numpy as np
import pytest

from bpd import config as c
from bpd import profile as pf

A = B = c.ALPHA_TK


@pytest.mark.parametrize("gain, expected", [(1500, 1.43), (2000, 1.84),
                                            (2500, 2.24), (3000, 2.63)])
def test_loss_aversion_matches_deck_page_11(gain, expected):
    assert pf.loss_aversion_from_g(gain, A, B) == pytest.approx(expected, abs=0.005)


def test_published_lambda_implies_the_decks_indifference_gain():
    assert pf.loss_aversion_from_g(2513, A, B) == pytest.approx(c.LAMBDA_TK, abs=0.005)


@pytest.mark.parametrize("p, gamma, expected", [
    (0.50, c.GAMMA_PLUS, 0.421), (0.50, c.GAMMA_MINUS, 0.454),
    (0.10, c.GAMMA_PLUS, 0.186), (0.90, c.GAMMA_PLUS, 0.712),
    (0.40, c.GAMMA_MINUS, 0.392), (0.60, c.GAMMA_MINUS, 0.518),
])
def test_weighting_matches_deck_pages_13_and_14(p, gamma, expected):
    assert pf.weight(p, gamma) == pytest.approx(expected, abs=0.0005)


def test_weighting_crosses_the_diagonal_where_the_deck_says():
    grid = np.linspace(0.05, 0.95, 90_001)
    crossing = grid[np.argmin(np.abs(pf.weight(grid, c.GAMMA_PLUS) - grid))]
    assert crossing == pytest.approx(0.339, abs=0.0005)


def test_value_function_matches_the_decks_worked_check():
    assert pf.value(-400, A, B, c.LAMBDA_TK) == pytest.approx(-438.5, abs=0.1)
    assert 0.5 * pf.value(-800, A, B, c.LAMBDA_TK) == pytest.approx(-403.5, abs=0.1)


def test_structured_note_reversal_reproduces_deck_page_22():
    v_bear = pf.value(-20_000, A, B, c.LAMBDA_TK)
    v_bull = pf.value(25_000, A, B, c.LAMBDA_TK)
    v_cap = pf.value(6_000, A, B, c.LAMBDA_TK)
    assert v_bear == pytest.approx(-13_712, abs=1)
    assert v_bull == pytest.approx(7_416, abs=1)
    assert v_cap == pytest.approx(2_112, abs=1)
    v_index = pf.weight(0.40, c.GAMMA_MINUS) * v_bear + pf.weight(0.60, c.GAMMA_PLUS) * v_bull
    v_note = pf.weight(0.60, c.GAMMA_PLUS) * v_cap
    assert v_index == pytest.approx(-1_856, abs=1)
    assert v_note == pytest.approx(1_001, abs=1)
    assert v_index < 0 < v_note


def test_symmetric_coin_flip_is_rejected_as_on_deck_page_11():
    value = 0.5 * (pf.value(1_000, A, B, c.LAMBDA_TK) + pf.value(-1_000, A, B, c.LAMBDA_TK))
    assert value == pytest.approx(-272.8, abs=0.1)


def test_curvature_is_one_when_the_certainty_equivalent_is_the_expected_value():
    assert pf.curvature_from_ce(500.0, 1_000.0) == pytest.approx(1.0, abs=1e-12)


def test_curvature_falls_as_the_client_demands_a_smaller_certain_amount():
    assert pf.curvature_from_ce(400.0, 1_000.0) < pf.curvature_from_ce(480.0, 1_000.0) < 1.0


def test_curvature_cancels_lambda_on_a_loss_only_lottery():
    prize, ce = 1_000.0, 400.0
    beta = pf.curvature_from_ce(ce, prize)
    for lam in (1.5, 2.25, 3.0):
        assert pf.value(-ce, A, beta, lam) == pytest.approx(
            0.5 * pf.value(-prize, A, beta, lam), rel=1e-12)


def test_risk_aversion_rises_with_loss_aversion_and_stays_inside_the_anchors():
    lo, hi = c.LAMBDA_RANGE
    assert pf.risk_aversion(lo) == pytest.approx(c.GAMMA_AT_LAMBDA_LOW)
    assert pf.risk_aversion(hi) == pytest.approx(c.GAMMA_AT_LAMBDA_HIGH)
    assert pf.risk_aversion(1.0) == pytest.approx(c.GAMMA_AT_LAMBDA_LOW)
    assert pf.risk_aversion(5.0) == pytest.approx(c.GAMMA_AT_LAMBDA_HIGH)
    assert pf.risk_aversion(2.25) > pf.risk_aversion(1.8)
