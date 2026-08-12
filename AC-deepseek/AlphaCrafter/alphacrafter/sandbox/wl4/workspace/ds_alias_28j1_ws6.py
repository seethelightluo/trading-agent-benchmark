import sys; sys.path.insert(0,'scripts')
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE
panels = load_panels(days=4000)
closes = close_panel(panels)
print("close panel shape:", closes.shape)
print("min date:", closes.index.min().date(), "max date:", closes.index.max().date())
print("n_assets:", len(closes.columns))
print("last 5 dates:", [d.date().isoformat() for d in closes.index[-5:]])
# missingness per year
print(closes.isna().mean().round(3))
