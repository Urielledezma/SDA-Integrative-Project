# Presentation plan: fifteen minutes and a live demo

> **Before you present, print the figures rather than retyping them:**
>
> ```bash
> python presentation_figures.py
> ```
>
> It computes the deterministic ones from the data and the profile, and reads the rest out
> of `results/`. It refuses to print the second group if the evaluation has not been run,
> if it was a smoke run, or if it was run for a different client than the code now
> produces, which is the failure that actually happened here when the loss-aversion
> estimator was corrected. A number spoken from memory that the notebook contradicts on
> screen is the worst thing that can happen in this room.

---

## The arc

One sentence to carry the whole talk: **the policy that scores best is the one that stops
being the portfolio we sold, and once that is fixed the entire case against discipline
turns out to be one decade's cross-section.**

---

## Minute by minute

### 1. The client and the question (1 min)

An individual investor, 100,000 pesos, ten years, a target of 400,000. Three questions in
order: what kind of risk taker are they, what portfolio does that imply, and what will
their own habits do to it.

### 2. Part 1, the profile (3 min)

Ten questions. Show the loss-aversion ladder, because it is the opening exercise of his
own prospect-theory lecture: a coin that loses 1,000 on heads, and the smallest gain at
which they would take it.

Say the identification order, because it shows we read the algebra rather than copied a
formula: **beta, then alpha, then lambda.** A loss-only lottery has lambda on both sides
of the indifference condition, so it cancels and leaves curvature alone. Only the last
position is forced; the two curvatures could be taken either way round.

**Then spend the rest of this slide on the appendix, because it is the strongest thing in
the report.** His deck's page 31 says the naive estimator is biased, because the two
decision weights are not both a half: 0.421 against 0.454. The corrected identity
multiplies by their ratio and comes out about 7% lower, and his own worked case is 2.08
instead of 2.24. **We use the corrected one, and our code reproduces both of his figures.**

If asked why not the formula from class: because he wrote the appendix that withdraws it.

Close with the limitation he states there: loss aversion and probability weighting are not
separately identified from one indifference, so our lambda is conditional on the weighting
function being the published one. Our ladder brackets the switch point; it does not cancel
the weights, so we do not claim that it does.

### 3. Part 2, the portfolio (2 min)

Mean-variance in the utility form, with cardinality, minimum lots and a cap per asset
class, solved with Gurobi.

Spend the slide on the input, not the output. Merton: the standard error of a mean depends
only on the calendar span. Chopra and Ziemba: errors in means cost ten times what errors
in covariances cost. Michaud: the optimiser prefers exactly the assets carrying positive
estimation error. So we shrink the mean toward its cross-sectional average on purpose,
which is what Jagannathan and Ma's result prescribes, rather than letting the long-only
constraint do it silently. Say plainly that we shrink the **mean** where his slide names a
covariance target or a prior, and that the departure is ours.

### 4. Part 3, the model (2 min)

The five elements on one slide. Dwell on two things only:

- **Why the index value is in the state.** A client sitting in cash has no drawdown of
  their own left to recover from, so the rule that brings them back must read the market.
- **The timing.** The trade is chosen before the month's return is known.

### 5. The policies (1 min)

Six rules with their classes: two baselines, the band as a PFA, the re-solving model as a
CFA, and the two behavioural rules, which are PFAs in form but are never tuned, because a
panic rule tuned to maximise wealth is no longer a panic rule.

### 6. The result, and the thing that went wrong (3 min)

**This is the centre of the talk. Do not rush it.**

Put up the main table. Holding wins the objective, and by a wide margin.

Then put up the drift column. Holding ends with roughly half the portfolio in a single
fund, against the five holdings and the 35% cap it was built under. Effective holdings
fall by about half.

Then say what the tuner did: searching the band against the objective alone picks the
widest value on the grid, where the rule never fires. **The policy search concluded that
the best rebalancing rule is not to rebalance**, which is the correct answer to a question
we had posed wrongly, because the cap was imposed once and the objective never asks for it
again.

Then say what we had to fix twice. Restricting the search to settings that keep the cap
turned out to be unsatisfiable: the optimiser places the portfolio within a point of the
cap, and weights move between the dates a policy acts, so the test rejected even a monthly
rebalance. What a policy can control is the **overshoot**, so the test bounds that instead,
at five percentage points, and we say the bound is ours.

### 7. The robustness test (2 min)

Give every fund the same expected return, which removes a cross-sectional spread of about
six percentage points a year and changes nothing else.

**The entire advantage of holding disappears.** The band becomes statistically
indistinguishable from it, and the re-solving model beats it outright. Panic selling is one
of only two policies that still lose significantly, the other being equal weights, and it
is the only one of the two that describes something a client would actually do.

So the case for doing nothing was built out of one decade's cross-section.

### 8. The recommendation (1 min)

Re-solve the portfolio every few months; keep the band as the simpler alternative for a
client who wants a rule they can check themselves. Discipline through re-optimisation costs
roughly a quarter of what discipline through a fixed band costs, and holds the portfolio
about as well.

Then the line worth ending on: **a client whose real panic threshold is 10% rather than 15%
loses more than the entire rebalancing decision is worth.** The adviser's effort belongs in
pre-committing them against selling.

### 9. Live demo (2 min)

`streamlit run app.py`, already running and warm before the talk starts.

1. Move the loss-aversion ladder up one rung and show the portfolio turn defensive.
2. Show the fan chart against the target line.
3. Show the cost of the tendencies.

Do not improvise beyond those three. Leave it on 200 scenarios; 500 is slow in front of a
room.

---

## Division of labour

| Section | Who |
|---|---|
| 1 and 2, client and profile, including the appendix | first speaker |
| 3 and 4, portfolio and model | second speaker |
| 5 and 6, policies and the two things we got wrong | third speaker |
| 7, 8 and 9, robustness, recommendation and demo | fourth speaker |

Everyone must be able to answer anything, because questions do not respect the split.

---

## Before the room

- Dashboard running and warm, on 200 scenarios.
- Notebook open at the main results table.
- `results/summary.txt` open in a second window, because every number asked about is in it.
- `preguntas-del-martes.md` read once that morning.

**The figures to know by heart.** Run `python presentation_figures.py` on the morning and
read them off it. The first group never changes unless the profile does, so it is filled
in here; the second comes from the evaluation and the script will refuse to print it until
the full run exists.

| What | Value |
|---|---|
| The client's loss aversion, uncorrected | 2.16 |
| The client's loss aversion, corrected | **2.00**, 7.3% lower |
| Risk aversion handed to Part 2 | 5.76 |
| The portfolio | 5 holdings, largest 34.1% against a 35% cap |
| Expected return and volatility | 12.08% and 9.47% a year |
| The cross-sectional spread the robustness test removes | **6.01% a year** |
| Median terminal wealth, holding | from the script |
| Median terminal wealth, the recommended policy | from the script |
| What discipline costs, with its interval | from the script |
| Largest single holding at the end, holding, worst scenario | from the script |
