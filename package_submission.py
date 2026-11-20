"""Build the zip that goes to Canvas, and refuse to build a broken one.

    python package_submission.py [--out submission.zip]

Canvas takes a single zip. This collects what the brief asks a team to submit, which is
all the code, a README with run instructions, `requirements.txt`, the report and the
dashboard, and it checks the things that would be embarrassing to discover afterwards:
that the report was actually executed, that it executed without errors, that the tests
pass, and that no credential file is being shipped.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent

INCLUDE = [
    "README.md", "LICENSE", "requirements.txt", "pytest.ini",
    ".editorconfig", ".gitattributes",
]
INCLUDE_TREES = ["projects"]

# The brief asks for one project and A is it. Anything else in projects/ is additional
# work, which is welcome in the zip when it is finished and must not hold up the
# submission when it is not.
SUBMISSION = "a-behavioral-portfolio"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
              ".playwright-cli", ".playwright", ".archives"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def is_wanted(path: pathlib.Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if path.suffix in SKIP_SUFFIX:
        return False
    return True


def check_notebook(path: pathlib.Path) -> list:
    """A report that was never executed is the failure worth catching here."""
    problems = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    executed = [c for c in code if c.get("outputs")]
    errors = [o for c in notebook["cells"] for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    if not code:
        problems.append(f"{path.name}: no code cells")
    elif len(executed) < len(code) - 1:
        problems.append(f"{path.name}: only {len(executed)} of {len(code)} code cells "
                        f"carry output, so it was not executed")
    if errors:
        problems.append(f"{path.name}: {len(errors)} cell(s) ended in an error")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run pytest first, for a quick rebuild")
    args = ap.parse_args()

    problems, files = [], []

    for name in INCLUDE:
        path = ROOT / name
        if path.exists():
            files.append(path)
        else:
            problems.append(f"missing {name}")

    for tree in INCLUDE_TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if path.is_file() and is_wanted(path):
                files.append(path)

    notebooks = [p for p in files if p.suffix == ".ipynb"]
    if not any(SUBMISSION in p.parts for p in notebooks):
        problems.append(f"no report notebook found for {SUBMISSION}, which is the submission")

    dropped = set()
    for path in notebooks:
        faults = check_notebook(path)
        if not faults:
            continue
        if SUBMISSION in path.parts:
            problems.extend(faults)
        else:
            # An unfinished extra is left out rather than shipped half-built, and the
            # exclusion is announced rather than done quietly.
            project_dir = next(p for p in path.parents if p.parent.name == "projects")
            dropped.add(project_dir)
            print(f"  LEAVING OUT {project_dir.name}: " + "; ".join(faults))
    if dropped:
        files = [f for f in files if not any(d in f.parents for d in dropped)]
        print(f"  the zip will contain {SUBMISSION} and nothing else from projects/")

    leaked = [p for p in files if p.name == ".env" or p.name.startswith(".env.")
              or p.name in {"kaggle.json", "credentials.json"}]
    problems.extend(f"credential file would be shipped: {p}" for p in leaked)

    # A smoke run writes the same filenames as the real evaluation, so the only thing
    # standing between a quick check and a shipped table is this. It applies to what is
    # being packaged, since a dropped project's tables are not going anywhere.
    for summary in ROOT.glob("projects/*/results/summary.txt"):
        if any(d in summary.parents for d in dropped):
            continue
        if "SMOKE RUN" in summary.read_text(encoding="utf-8"):
            problems.append(f"{summary.relative_to(ROOT).as_posix()} is from a smoke run, "
                            f"so results/ holds numbers too small to report")

    print("checking the reports against what they computed")
    verify = subprocess.run(
        [sys.executable, "verify_report.py",
         str(ROOT / "projects" / SUBMISSION / "A_behavioral_portfolio.ipynb")],
        cwd=ROOT, capture_output=True, text=True)
    for line in (verify.stdout or "").strip().splitlines():
        print("  " + line)
    if verify.returncode != 0:
        problems.append("a report quotes figures its own cells did not print, "
                        "or was never executed; see verify_report.py above")

    for script in sorted((ROOT / "projects" / SUBMISSION).glob("presentation_figures.py")):
        check = subprocess.run([sys.executable, script.name], cwd=script.parent,
                               capture_output=True, text=True)
        if check.returncode != 0:
            tail = [l for l in (check.stdout or "").splitlines() if "NOT AVAILABLE" in l]
            problems.append(f"{script.parent.name}: " +
                            (tail[0].strip() if tail else "its results are not usable"))

    if not args.skip_tests:
        print("running the tests")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                                cwd=ROOT, capture_output=True, text=True)
        tail = (result.stdout or result.stderr).strip().splitlines()
        print("  " + (tail[-1] if tail else "no output"))
        if result.returncode != 0:
            problems.append("the test suite does not pass")

    if problems:
        print("\nNOT PACKAGING. Fix these first:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    out = ROOT / args.out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())

    size = out.stat().st_size / 1e6
    print(f"\nwrote {out.name}: {len(files)} files, {size:.1f} MB")
    # Report what went in, not what was considered. An earlier version printed the
    # notebooks it had checked, which included one it had just excluded.
    for path in [f for f in files if f.suffix == ".ipynb"]:
        print(f"  report included and executed: {path.relative_to(ROOT).as_posix()}")
    for folder in sorted(dropped):
        print(f"  left out, unfinished:         {folder.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
