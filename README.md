# Sequential Decision Analytics: Integrative Project

Two sequential decision problems modelled with Powell's framework, each solved with a
baseline and at least two policies from different policy classes, each evaluated in its own
simulator on common random numbers, and each delivered with a dashboard a client can drive
without reading the code.

Coursework for **Comportamiento en las Finanzas y Toma de Decisiones**, Financial
Engineering programme, ITESO — Universidad Jesuita de Guadalajara.

**Team:** Alan Jesús Hernández Soto · Francisco Uriel Ledezma Chávez · Esteban Vega Campos · Santiago Villegas Baltazar  
**Professor:** Luis Felipe Gómez Estrada  
**Term:** Autumn 2026 · 5th semester

---

## What to read first

The brief asks each team for one project, and **Project A is ours**, so the report, the
dashboard and the presentation to look at are the ones in
[`projects/a-behavioral-portfolio/`](projects/a-behavioral-portfolio). Project B was built
afterwards, on the same framework and to the same evaluation standard, because the team
wanted the revenue-management problem as well, and it is included here rather than
withheld. Nothing in A depends on it.

| Project | Client | Decision each period | Folder |
|---|---|---|---|
| **A. Behavioral portfolio dashboard** | Individual investor | The trades made each month | [`projects/a-behavioral-portfolio/`](projects/a-behavioral-portfolio) |
| B. Dynamic ticket pricing | Event promoter | Today's price from the grid | [`projects/b-dynamic-pricing/`](projects/b-dynamic-pricing) |
| C. University course scheduling | Department head | Weekly timetable adjustments | not attempted |

**Project C was not attempted, and the reason belongs here rather than in a footnote.** The
brief describes its department data and its enrollment generator as provided, and neither
exists: the course materials contain no such files. Building it would have meant inventing
twenty five courses, thirty seven sections, fourteen professors, ten rooms and five years of
enrollment history, and then reporting results measured against a world we had made up. With
one project graded and four days to the deadline, the team put the time into A instead.

It is deferred rather than abandoned. The questions that would have to be answered before
any of its data could be written down are listed in the team's notes, and the first of them
is the real curriculum, because a timetable built from invented course names is an exercise
rather than a department.

---

## Running it

**Requirements:** Python 3.12. The solver is Gurobi, which the brief prescribes for the
portfolio model, and the restricted licence that ships with `gurobipy` is large enough for
every model here, so nothing has to be registered to reproduce the results.

```bash
pip install -r requirements.txt
python -m pytest -q                 # the properties each report's conclusions rest on
```

The dashboard for Project A, which is the one the brief asks for, runs with one command:

```bash
streamlit run projects/a-behavioral-portfolio/app.py
```

Each project's own README gives the commands that regenerate its tables and re-execute its
report from a clean kernel.

Everything that has to be run before submitting is one command, which runs the stages
strictly in order and builds the zip at the end:

```bash
python finish_submission.py          # Project A, which is the submission
python finish_submission.py --all    # A and B
```

It takes about fifteen minutes per project and nothing is backgrounded, because two
executions of one notebook were once started in parallel here and the older one finished
last, overwriting the newer.

The packaging step is also available on its own, and it refuses to build a zip if a report
notebook was never executed, if any cell ended in an error, if a report quotes figures its
own cells did not print, if `results/` was left by a smoke run or by a different client
than the code now produces, if the tests do not pass, or if a credential file would be
shipped:

```bash
python package_submission.py
```

---

## Conventions

- Code, documentation and commit messages in English.
- A report's prose is edited in its `build_notebook.py`, never in the `.ipynb`, which is
  overwritten on every build.
- Parameters and seeds live in the project's own `config.py`, never inline in a report.
- Tuning and final evaluation use **separate seeds**, and every policy in a comparison sees
  **the same scenarios**, because differences measured on different draws mix the effect of
  the policy with the effect of the draw.
- No project imports from another, so each one can be read and defended on its own.

---

## Licence

MIT, see [LICENSE](LICENSE). Prices are retrieved with `yfinance` for academic use and the
cached copy in each project's `data/` folder exists so the results reproduce offline.
