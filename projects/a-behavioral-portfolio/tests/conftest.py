import pathlib
import sys

# The project's library is imported from its src/ rather than installed, as the
# notebook and the dashboard do.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
