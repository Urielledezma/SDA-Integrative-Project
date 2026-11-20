# Sequential Decision Analytics: Integrative Project

Three sequential decision problems modelled with Powell's framework, each solved with a
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
[`projects/a-behavioral-portfolio/`](projects/a-behavioral-portfolio). Projects B and C were
built afterwards, on the same framework and to the same evaluation standard, because the
team wanted the other two problem classes as well, and they are included here rather than
withheld. Nothing in A depends on them.

| Project | Client | Decision each period | Folder |
|---|---|---|---|
| **A. Behavioral portfolio dashboard** | Individual investor | The trades made each month | [`projects/a-behavioral-portfolio/`](projects/a-behavioral-portfolio) |
| B. Dynamic ticket pricing | Event promoter | Today's price from the grid | `projects/b-dynamic-pricing/` |
| C. University course scheduling | Department head | Weekly timetable adjustments | `projects/c-course-scheduling/` |

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
