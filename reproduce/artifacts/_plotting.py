"""Shared plotting setup: headless matplotlib and the analysis plot helpers."""
import os
import sys
import tempfile

from reproduce.paths import REPO

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="cafe-mpl-"))
if str(REPO / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "analysis"))

import coreness_plots  # noqa: E402,F401
import plot_precision_eg_tradeoff_table_match as pareto_style  # noqa: E402,F401
