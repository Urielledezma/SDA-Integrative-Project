# Presentation plan: fifteen minutes and a live demo

Every figure below is in `results/summary.txt` and in the executed report, so nothing here
has to be remembered separately from the work.

---

## The arc

One sentence to carry the whole talk: **the policy that scores best is the one that stops
being the portfolio we sold, and once that is fixed the entire case against rebalancing
turns out to be one decade's cross-section.**

---

## Minute by minute

### 1. The client and the question (1 min)

An individual investor, 100,000 to invest, ten years, and a target of 400,000. Three
questions in order: what kind of risk taker are they, what portfolio does that imply, and
what will their own habits do to it.

### 2. Part 1, the profile (2 min)

Ten questions. Show the loss-aversion ladder, because it is the opening exercise of the
prospect-theory lecture: a coin that loses 1,000 on heads, and the smallest gain at which
they would take it. This client switches at 2,600, which brackets to 2,400, which gives
**lambda 2.16**.

Say the identification order out loud, because it is the part that shows we understood the
algebra rather than copied a formula: **beta, then alpha, then lambda.** A loss-only lottery
has lambda on both sides of the indifference condition, so it cancels and leaves curvature
alone. Only then does the mixed lottery identify lambda.

One line on validation: the implementation reproduces fifteen numbers printed in the course
deck, including w+(0.5) = 0.421 and the crossing of the diagonal at p = 0.339.

### 3. Part 2, the portfolio (2 min)

Mean-variance in the utility form, with cardinality, minimum lots and a cap of 40% per asset
class, solved with Gurobi. Five holdings, 11.87% a year expected, 9.11% volatility.

Spend most of this slide on the input, not the output. Merton says the standard error of a
mean depends only on the calendar span. Chopra and Ziemba put the damage from errors in
means at ten times that of covariances. Michaud says the optimiser prefers exactly the
assets carrying positive estimation error. So we shrink the mean toward its cross-sectional
average on purpose, which is what Jagannathan and Ma's result prescribes, rather than
letting the long-only constraint do it silently.

### 4. Part 3, the model (2 min)

The five elements on one slide. Dwell on two things only:

- **Why the index value is in the state.** A client sitting in cash has no drawdown of their
  own left to recover from, so the rule that brings them back has to read the market.
- **The timing.** The trade is chosen before the month's return is known.

### 5. The policies (2 min)

Six rules in a table with their classes: two baselines, the band as a PFA, the re-solving
model as a CFA, and the two behavioural rules, which are PFAs in structure but are never
tuned, because a panic rule tuned to maximise wealth is no longer a panic rule.

### 6. The result, and the thing that went wrong (3 min)

**This is the centre of the talk. Do not rush it.**

Put up the main table. Holding wins: 15,771 against the band's −12,598, and a median of
471,725 against 385,631.

Then put up the drift column. Holding ends with **59% in one fund on average and 88% in the
worst scenario**, against the 35% cap it was built under. Effective holdings fall from 4.4
to 2.3.

Then say what the tuner did: searching the band against the objective alone picks **50%**,
where the rule never fires. The policy search concluded that the best rebalancing rule is
not to rebalance, which is the correct answer to a question we had posed wrongly, because
the cap was imposed once and the objective never asks for it again.

The fix is to search only over settings that keep the cap at the 95th percentile. That gives
**5%**, and the honest price is 28,370 of objective and about 86,000 of median wealth.

### 7. The robustness test (2 min)

Give every fund the same expected return, which removes a cross-sectional spread of 6.01% a
year and changes nothing else. The 28,370 **disappears**: the band is now 265 ahead with an
interval of [−564, 1,095], indistinguishable.

So the case for doing nothing was built out of one decade's cross-section. Panic selling is
the only policy that still loses significantly, at −970 with [−1,467, −474], because its cost
is a property of the rule and not of the sample.

### 8. The recommendation (1 min)

Rebalance on a 5% band. It is the only admissible policy that keeps the mandate, its cost
vanishes under the robustness test, and holding's 88% worst case cannot be put in front of
the client who was profiled for something else.

Then the line worth ending on: **a client whose real panic threshold is 10% rather than 15%
loses about 141,000, which is more than the entire rebalancing decision is worth.** The
adviser's effort belongs in pre-committing them against selling.

### 9. Live demo (2 min)

`streamlit run app.py`, already running before the talk starts.

1. Move the loss-aversion ladder up one rung and show the portfolio turn defensive.
2. Show the fan chart against the target line.
3. Show the cost of the tendencies.

Do not improvise beyond those three. Every slider recomputes, and a 500-scenario setting
takes time in front of an audience, so leave it at 200.

---

## Division of labour

Four speakers, and whoever is not speaking watches the clock.

| Section | Who |
|---|---|
| 1 and 2, client and profile | first speaker |
| 3 and 4, portfolio and model | second speaker |
| 5 and 6, policies and results | third speaker |
| 7, 8 and 9, robustness, recommendation and demo | fourth speaker |

Everyone must be able to answer anything, because questions do not respect the split.

---

## Before the room

- Dashboard running and warm, on 200 scenarios.
- Notebook open at the main results table.
- `results/summary.txt` open in a second window, because every number asked about is in it.
- Know these six by heart: **2.16, 471,725, 385,631, 28,370, 6.01%, 88%.**
