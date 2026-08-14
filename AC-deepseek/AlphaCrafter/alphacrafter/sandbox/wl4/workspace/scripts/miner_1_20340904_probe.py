"""miner_1 (2034-09-04) probe: verify data availability and load timing."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd

t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] importing...", flush=True)
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

t1 = time.time()
print(f"[{t1-t0:5.1f}s] loading panels (days=4000)...", flush=True)
panels = load_panels(days=4000)
t2 = time.time()
print(f"[{t2-t0:5.1f}s] loaded {len(panels)} panels in {t2-t1:.1f}s", flush=True)

closes = close_panel(panels)
print(f"closes shape={closes.shape}", flush=True)
print(f"date range: {closes.index.min().date()} -> {closes.index.max().date()}", flush=True)
print(f"n tradable loaded: {len(closes.columns)}", flush=True)
missing = [a for a in TRADABLE if a not in closes.columns]
print(f"missing tradable: {missing}", flush=True)
for a in TRADABLE:
    n = len(panels[a]) if a in panels else 0
    print(f"  {a:10s} rows={n}", flush=True)
print(f"[{time.time()-t0:5.1f}s] probe done", flush=True)
