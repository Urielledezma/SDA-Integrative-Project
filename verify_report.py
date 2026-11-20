"""Check that every figure a report's prose quotes appears in what the report computed.

    python verify_report.py [notebook.ipynb ...]

With no arguments it checks every executed report in projects/.

A report is written in two places: the markdown cells assert things, and the code cells
produce the numbers those assertions rest on. Nothing stops the two from drifting apart,
and in this project they did, twice: once when the notebook tuned without a constraint the
driver tuned with, and once when a figure was carried over from a superseded run. Both
were caught by hand. This does it on demand instead.

The check is deliberately crude. It pulls every number out of the markdown and asks
whether that number, allowing for the usual formatting, appears anywhere in the printed
output of the same notebook. A number that does not is not necessarily wrong, because it
may be arithmetic on two printed figures or a value quoted from the course, so the script
reports rather than fails on those and leaves the judgement to a reader. What it is good
at is catching the figure that no longer exists anywhere in the run.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Numbers that are the course's rather than the run's, so they are expected to appear in
# the prose without being computed. Each needs a reason, and an unexplained entry here is
# how a check like this quietly stops checking.
FROM_THE_COURSE = {
    "0.88": "Tversky and Kahneman curvature, deck 003 page 8",
    "2.25": "Tversky and Kahneman loss aversion, deck 003 page 8",
    "0.421": "w+(0.5), deck 003 page 14",
    "0.454": "w-(0.5), deck 003 page 14",
    "2.24": "the deck's own uncorrected lambda at G = 2500, page 11",
    "2.08": "the appendix's corrected lambda at the same G, page 31",
    "1.4": "the low end of the published lambda range, deck 003 page 24",
    "3.0": "the high end of the same range",
    "10": "the Chopra and Ziemba ratio, deck 002 page 20",
    "35": "the position cap, a parameter of ours in config",
    "40": "the class cap, likewise",
    "0.5": "the shrinkage intensity, likewise",
    "5": "the drift tolerance in percentage points, likewise",
    "500": "the number of evaluation scenarios, from the brief",
    "100": "the number of tuning scenarios, from config",
    "129": "months of history, printed by the setup cell",
    "25": "funds in the universe",
    "12": "months in a year, and the re-solve interval",
    "60": "the selling window in days, Project B",
    "2000": "the venue capacity, Project B",
}

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)(?![\w])")


def cells(notebook, kind):
    return [c for c in notebook["cells"] if c["cell_type"] == kind]


def printed_text(notebook) -> str:
    out = []
    for cell in notebook["cells"]:
        for item in cell.get("outputs", []):
            if item.get("output_type") == "stream":
                out.append("".join(item.get("text", "")))
            data = item.get("data", {})
            if "text/plain" in data:
                out.append("".join(data["text/plain"]))
    return "\n".join(out)


def appears(value: str, haystack: str) -> bool:
    """Allow for the formatting difference between prose and a printed table."""
    plain = value.replace(",", "")
    candidates = {value, plain}
    if "." not in plain:
        candidates |= {f"{plain}.0", f"{plain}.00"}
        try:
            candidates.add(f"{int(plain):,}")
        except ValueError:
            pass
    else:
        whole, _, frac = plain.partition(".")
        candidates.add(whole)
        try:
            candidates.add(f"{int(whole):,}")
        except ValueError:
            pass
        candidates.add(plain.rstrip("0").rstrip("."))
    return any(c and c in haystack for c in candidates)


def check(path: pathlib.Path) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = cells(notebook, "code")
    executed = [c for c in code if c.get("outputs")]
    errors = [o for c in notebook["cells"] for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    output = printed_text(notebook)

    try:
        shown = path.relative_to(ROOT).as_posix()
    except ValueError:
        shown = path.as_posix()
    print(f"\n=== {shown}")
    print(f"  {len(notebook['cells'])} cells, {len(executed)} of {len(code)} code cells "
          f"with output, {len(errors)} errors")
    if not executed:
        print("  NOT EXECUTED, so there is nothing to check the prose against")
        return 1
    if errors:
        print("  a cell ended in an error, so the outputs cannot be trusted")

    untraced = {}
    for cell in cells(notebook, "markdown"):
        source = "".join(cell["source"])
        # The bibliography is years, volumes and page ranges, none of which the run
        # computes, so checking it would report nothing but noise.
        if re.search(r"^#+\s*\d*\.?\s*References\b", source, re.MULTILINE):
            continue
        for value in NUMBER.findall(source):
            if value in FROM_THE_COURSE or appears(value, output):
                continue
            # A bare four-digit number in this range is a year in a citation.
            if re.fullmatch(r"(19|20)\d\d", value):
                continue
            untraced.setdefault(value, 0)
            untraced[value] += 1

    small = {v for v in untraced if "." not in v and "," not in v and len(v) <= 2}
    interesting = {v: n for v, n in untraced.items() if v not in small}

    if not interesting:
        print("  every figure in the prose appears in the printed output")
        return 0 if not errors else 1
    print(f"  {len(interesting)} figure(s) in the prose that no cell printed:")
    for value, count in sorted(interesting.items(), key=lambda kv: -kv[1]):
        print(f"    {value:>14}  quoted {count}x")
    print("  Each is either arithmetic on two printed numbers, a value taken from the")
    print("  course, or a figure left over from a superseded run. Check which.")
    return 1


def main():
    targets = [pathlib.Path(a) for a in sys.argv[1:]]
    if not targets:
        targets = sorted((ROOT / "projects").rglob("*.ipynb"))
        targets = [t for t in targets if ".ipynb_checkpoints" not in t.parts]
    if not targets:
        print("no notebooks found")
        return 1
    return max(check(t) for t in targets)


if __name__ == "__main__":
    sys.exit(main())
