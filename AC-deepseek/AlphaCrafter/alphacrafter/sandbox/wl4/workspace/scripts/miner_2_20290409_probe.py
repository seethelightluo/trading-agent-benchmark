"""miner_2 probe: verify data window available through API (no lookahead)."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

panels = load_panels(days=4000)
closes = close_panel(panels)
print("close panel shape:", closes.shape)
print("min date:", closes.index.min().date(), "max date:", closes.index.max().date())
print("assets loaded:", list(closes.columns))
print("last row:")
print(closes.tail(3))
# check missingness in last 30 days
print("NaN count per asset last 30d:")
print(closes.tail(30).isna().sum())
