"""The instrument must recover the parameters it was built to recover."""
import numpy as np
import pytest

from bpd import config as c
from bpd import questionnaire as q


def test_the_instrument_has_ten_questions_with_unique_ids():
    assert len(q.QUESTIONS) == 10
    assert len({item["id"] for item in q.QUESTIONS}) == 10


def test_every_question_is_answered_by_the_default_client():
    assert {item["id"] for item in q.QUESTIONS} == set(q.default_answers())


def test_the_quoted_client_recovers_the_published_curvatures():
    p = q.score(q.default_answers())
    assert p.alpha == pytest.approx(c.ALPHA_TK, abs=0.001)
    assert p.beta == pytest.approx(c.BETA_TK, abs=0.001)


def test_the_quoted_clients_loss_aversion_is_the_corrected_ladder_midpoint():
    from bpd import profile
    p = q.score(q.default_answers())
    ratio = profile.weight(0.5, c.GAMMA_PLUS) / profile.weight(0.5, c.GAMMA_MINUS)
    assert p.lam == pytest.approx(ratio * (2_400.0 / 1_000.0) ** c.ALPHA_TK, abs=0.001)
    assert 1.9 < p.lam < 2.1


def test_question_five_offers_only_references_the_model_implements():
    from bpd import simulator
    q5 = next(item for item in q.QUESTIONS if item["id"] == "q5_reference")
    for option in q5["options"]:
        answers = dict(q.default_answers())
        answers["q5_reference"] = option
        reference = simulator.reference_wealth(option, 100_000.0)
        assert reference > 0
    assert len(set(simulator.reference_wealth(o, 100_000.0) for o in q5["options"])) == len(q5["options"])


def test_a_recovered_curvature_round_trips_for_any_alpha():
    for alpha in (0.6, 0.75, 0.88, 1.0):
        answers = dict(q.default_answers())
        answers["q2_gain_ce"] = q.GAIN_PRIZE * 0.5 ** (1.0 / alpha)
        answers["q4_gain_ce_small"] = q.GAIN_PRIZE_SMALL * 0.5 ** (1.0 / alpha)
        answers["q3_loss_ce"] = q.LOSS_PRIZE * 0.5 ** (1.0 / alpha)
        p = q.score(answers)
        assert p.alpha == pytest.approx(alpha, abs=0.001)
        assert p.beta == pytest.approx(alpha, abs=0.001)


def test_a_higher_rung_on_the_ladder_means_more_loss_aversion():
    lams = []
    for rung in q.LADDER:
        answers = dict(q.default_answers())
        answers["q1_ladder"] = rung
        lams.append(q.score(answers).lam)
    assert lams == sorted(lams)
    assert all(b > a for a, b in zip(lams, lams[1:]))


def test_risk_aversion_is_monotone_in_the_ladder_and_bounded_by_the_anchors():
    gammas = []
    for rung in q.LADDER:
        answers = dict(q.default_answers())
        answers["q1_ladder"] = rung
        gammas.append(q.score(answers).gamma)
    assert gammas == sorted(gammas)
    assert min(gammas) >= c.GAMMA_AT_LAMBDA_LOW - 1e-9
    assert max(gammas) <= c.GAMMA_AT_LAMBDA_HIGH + 1e-9


def test_the_disciplined_client_raises_no_behavioural_flag():
    answers = dict(q.default_answers())
    answers.update({"q6_drawdown": "Nothing, I keep the plan",
                    "q7_chasing": "Nothing, I keep the plan",
                    "q8_disposition": "Whichever the plan says",
                    "q9_monitoring": "Once or twice a year",
                    "q10_anchor": "Sell it and follow the plan"})
    p = q.score(answers)
    assert not any(p.flags.values())
    assert not p.panic_selling and not p.performance_chasing


def test_the_quoted_client_raises_every_behavioural_flag():
    p = q.score(q.default_answers())
    assert all(p.flags.values())
    assert p.panic_selling and p.performance_chasing


def test_inconsistent_gain_answers_are_reported_rather_than_averaged_silently():
    answers = dict(q.default_answers())
    answers["q4_gain_ce_small"] = q.GAIN_PRIZE_SMALL * 0.5 ** (1.0 / 0.55)
    p = q.score(answers)
    assert any("differ by more than" in note for note in p.notes)


def test_a_loss_aversion_outside_the_published_range_is_flagged_not_overwritten():
    answers = dict(q.default_answers())
    answers["q1_ladder"] = q.LADDER[0]
    p = q.score(answers)
    assert p.lam < c.LAMBDA_RANGE[0]
    assert any("outside the 1.4 to 3.0 range" in note for note in p.notes)
    assert p.gamma == pytest.approx(c.GAMMA_AT_LAMBDA_LOW)
