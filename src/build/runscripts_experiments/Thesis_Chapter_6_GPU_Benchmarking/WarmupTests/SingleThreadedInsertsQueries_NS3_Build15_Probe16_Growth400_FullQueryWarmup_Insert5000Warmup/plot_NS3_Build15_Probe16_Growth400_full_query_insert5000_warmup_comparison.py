#!/usr/bin/env python3
"""Plot the NS3, B15/P16, growth-400 enhanced warm-up experiment."""

from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = (
    HERE.parent
    / "SingleThreadedInsertsQueries"
    / "plot_st_insert_st_query_warmup_comparison.py"
)

plotter = runpy.run_path(str(BASE_SCRIPT), run_name="growth400_warmup_plot_base")
plot_globals = plotter["main"].__globals__
plot_globals["HERE"] = HERE
plot_globals["NO_WARMUP_CSV"] = next(
    HERE.glob("*Growth400_NS3*_NoWarmups.csv")
)
plot_globals["WARMUP_CSV"] = next(
    HERE.glob("*Growth400_NS3*_Warmups.csv")
)
plotter["main"]()

old_stem = HERE / "st_inserts_st_queries_warmup_vs_no_warmup_all_operations_linear"
new_stem = HERE / "st_inserts_st_queries_NS3_Build15_Probe16_Growth400_full_query_insert5000_warmup_comparison"
for suffix in (".png", ".pdf"):
    old_stem.with_suffix(suffix).replace(new_stem.with_suffix(suffix))

print(f"Created {new_stem.with_suffix('.png')}")
print(f"Created {new_stem.with_suffix('.pdf')}")
