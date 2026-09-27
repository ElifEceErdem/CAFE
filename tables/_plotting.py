"""Shared plotting setup: headless matplotlib and the plot helpers of Figures 1, 3, 4."""
import os
import sys
import tempfile

from tables._common import REPO

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="cafe-mpl-"))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import coreness_plots  # noqa: E402,F401  (experiments/coreness_plots.py, shared with the runners)
from tables import _pareto_style as pareto_style  # noqa: E402,F401
