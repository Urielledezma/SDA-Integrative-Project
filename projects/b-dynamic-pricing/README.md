# Project B — Dynamic ticket pricing

**This is additional work.** Project A is the team's submission for the brief. B was built
afterwards, on the same framework and to the same evidential standard, because the team
wanted the revenue-management problem as well.

A promoter has 2,000 seats and 60 days. A seat unsold on the day is worth nothing, the
price moves once a day within a fixed grid, and buyers resent paying more than the price
they remember. Three policies are compared: the best single fixed price, a published
schedule that reacts to how fast the seats are going, and a daily re-plan that solves a
mixed-integer programme over the remaining days and charges only today's price.

---

## Contents

| File | What it is |
|---|---|
| [`B_dynamic_pricing.ipynb`](B_dynamic_pricing.ipynb) | The report |
| [`build_notebook.py`](build_notebook.py) | Assembles the report. **The notebook's source of truth** |
| [`app.py`](app.py) | The promoter's dashboard |
| [`run_analysis.py`](run_analysis.py) | Writes every table without opening the notebook |
| [`calibrate.py`](calibrate.py) | Fits the arrival curve to the Kaggle flight-prices data |
| [`src/dtp/`](src/dtp) | Configuration, the demand simulator, the policies |
| [`results/`](results) | Tables as CSV, regenerated on every run |

---

## Running it

```bash
streamlit run app.py
python run_analysis.py            # about fifteen minutes
python run_analysis.py --quick    # smoke test, about seven minutes
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace B_dynamic_pricing.ipynb
```

---

## The calibration was written and not run

The brief asks for the shape of the arrival curve to be re-estimated from the Kaggle
flight-prices dataset. `calibrate.py` does that: it reads the file in chunks, keeps one
route, turns the panel into tickets sold per day to departure and fits
`lambda(u) = A exp(B u)` by least squares on the log.

It was not run, and the report says so in section 4 rather than implying otherwise. The
file is 31 GB, Project A is the graded submission, and the team judged the download a poor
use of the four days before the deadline on work that is already additional. The arrival
parameters in `src/dtp/config.py` are therefore ours, set so that expected arrivals over
the window are about 1.4 times capacity, and the sensitivity section shows which
conclusions would survive a different curve.

One thing the script checks before fitting, which matters if it is ever run: the
`seatsRemaining` column in that dataset is capped at ten, so it counts seats left at a
quoted fare rather than seats on the aircraft. The shape of its decline still says when
demand arrives while its level is not a ticket count, so the script rescales the level to
the venue and records that it did.
