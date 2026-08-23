"""Probe: confirm data availability, panel date range, and current sim date for factor research."""
from factor_validation_lib import load_panel, TRADABLE

panel = load_panel()
print("panel shape:", panel.shape)
print("panel date range:", panel.index.min(), "->", panel.index.max())
print("n assets with data:", panel.notna().sum(axis=1).ge(1).sum(), "dates")

# final 5 rows coverage
print("final rows (tradable count):")
for d in panel.index[-5:]:
    n = panel.loc[d].notna().sum()
    print(" ", d.date(), "n=", n)