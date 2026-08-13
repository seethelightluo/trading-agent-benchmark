import sys
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel
panels = load_panels(days=4000)
closes = close_panel(panels)
print("closes shape:", closes.shape)
print("date range:", closes.index.min().date(), "->", closes.index.max().date())
print("last 3 rows:")
print(closes.tail(3))
