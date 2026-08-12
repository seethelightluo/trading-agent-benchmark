import sys
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel
panels = load_panels(days=3000)
px = close_panel(panels)
print("assets:", len(px.columns), "dates:", len(px))
print("date range:", px.index.min().date(), "->", px.index.max().date())
print("last 3 dates:", [str(d.date()) for d in px.index[-3:]])
print("volume coverage:", {a: (panels[a]['volume'].notna().mean() if 'volume' in panels[a] else 0) for a in px.columns})
