"""Run everything that has to be run, in order, and build the zip.

    python finish_submission.py              # Project A only, which is the submission
    python finish_submission.py --all        # A and B

One command, one process at a time. Two executions of the same notebook were once started
in parallel here and the older one finished last, overwriting the newer, so the stages
below are strictly sequential and nothing is backgrounded.

Each stage prints how much memory is free before it starts, because an earlier run of this
work was stopped by the operating system at six percent free, and the warning is cheaper
than discovering it sixteen minutes in.
"""
import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent


def free_memory_gb():
    """Best effort. Returns None where it cannot be measured rather than guessing."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip()) / 1024 / 1024
    except Exception:
        return None


def stage(title, command, cwd, force=False):
    free = free_memory_gb()
    note = f"{free:.1f} GB free" if free is not None else "memory unknown"
    print(f"\n=== {title}   [{note}]", flush=True)
    if free is not None and free < 1.5 and not force:
        print("  Less than 1.5 GB free. A run of this length was stopped by the system")
        print("  once at that level, so close some windows and start again, or pass")
        print("  --yes to go ahead anyway.")
        if not sys.stdin.isatty():
            # Launched without a terminal, so there is nobody to answer a prompt and
            # asking one would only raise. Stopping here is the honest outcome.
            return False
        if input("  Continue anyway? [y/N] ").strip().lower() != "y":
            return False
    started = time.time()
    result = subprocess.run(command, cwd=cwd)

    print(f"  {'done' if result.returncode == 0 else 'FAILED'} "
          f"in {time.time() - started:.0f} s")
    return result.returncode == 0


def project(folder, notebook, force=False):
    where = ROOT / "projects" / folder
    steps = [
        (f"{folder}: the tables", [sys.executable, "run_analysis.py"]),
        (f"{folder}: build the report", [sys.executable, "build_notebook.py"]),
        (f"{folder}: execute the report",
         [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
          "--execute", "--inplace", notebook]),
    ]
    for title, command in steps:
        if not stage(title, command, where, force):
            print(f"\nStopped at: {title}. Nothing after it has run.")
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="also run Project B, which is additional work")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="go ahead even when memory is below the safe line")
    args = ap.parse_args()

    print("This takes roughly fifteen minutes for A, and about the same again for B.")
    print("Nothing is backgrounded, so leave the window alone until it finishes.")

    if not project("a-behavioral-portfolio", "A_behavioral_portfolio.ipynb", args.yes):
        return 1
    if args.all and not project("b-dynamic-pricing", "B_dynamic_pricing.ipynb", args.yes):
        return 1

    if not stage("the figures for the presentation",
                 [sys.executable, "presentation_figures.py"],
                 ROOT / "projects" / "a-behavioral-portfolio", force=True):
        print("\nThe reports ran but their figures did not check out. Read the message "
              "above before packaging.")
        return 1

    if args.no_zip:
        return 0
    if not stage("the submission zip", [sys.executable, "package_submission.py"],
                 ROOT, force=True):
        print("\nThe reports ran but the zip was refused. The reasons are above, and "
              "each one is a real problem rather than a formality.")
        return 1

    print("\nDone. The zip is submission.zip, and the figures to memorise are printed "
          "above the packaging step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
