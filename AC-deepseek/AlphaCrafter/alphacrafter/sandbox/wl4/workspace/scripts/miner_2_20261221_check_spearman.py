"""miner_2 2026-12-21: spearman-rho diagnostic for batch5 gate-passing candidates
vs the 4 effective library factors (provenance for the deterministic pairwise gate)."""
from __future__ import annotations
import sys, json, base64, zlib, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

END = pd.Timestamp("2026-12-18")
panels = load_panels(3000)
closes_all = close_panel(panels)
idx = (closes_all.index >= pd.Timestamp("2020-01-01")) & (closes_all.index <= END)
closes = closes_all.loc[idx]
clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}

# max_drawdown_60d
def max_dd(s):
    roll_max = s.rolling(60).max()
    dd = s / roll_max - 1.0
    return dd.rolling(60).min()
mdd = pd.DataFrame({a: max_dd(s) for a, s in clean.items()}).reindex(closes_all.index).loc[idx]

# tech_beta_spread_60d
ndx_ret = clean["NDX"].pct_change()
spx_ret = clean["SPX"].pct_change()
def rolling_beta(ar, dr, win=60):
    z = pd.concat([ar.rename("a"), dr.rename("m")], axis=1).dropna()
    return (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var())
def tech_spread(s):
    return rolling_beta(s.pct_change(), ndx_ret) - rolling_beta(s.pct_change(), spx_ret)
tbs = pd.DataFrame({a: tech_spread(s) for a, s in clean.items()}).reindex(closes_all.index).loc[idx]

# library factors
lib = {}
for p in sorted(Path("factors").glob("*.json")):
    if p.name == "factor_ensemble.json":
        continue
    d = json.loads(p.read_text())
    sa = d.get("validation", {}).get("signal_artifact")
    if not sa:
        continue
    fmt = sa.get("format")
    if fmt == "base64:zlib:csv":
        raw = zlib.decompress(base64.b64decode(sa["data"]))
        df = pd.read_csv(io.BytesIO(raw), index_col=0)
        df.index = pd.to_datetime(df.index)
    elif fmt == "panel_json_v1":
        df = pd.DataFrame(sa["values"], index=pd.to_datetime(sa["dates"]), columns=sa["assets"])
    else:
        continue
    lib[d["factor_id"]] = df.reindex(closes.index)

def spearman_pair(cand, lib_df):
    both = pd.concat([cand.stack().rename("c"), lib_df.stack().rename("l")], axis=1).dropna()
    if len(both) < 30:
        return None
    return float(both["c"].corr(both["l"], method="spearman"))

for name, cand in [("max_drawdown_60d", mdd), ("tech_beta_spread_60d", tbs)]:
    print(f"== {name} ==")
    for k, v in lib.items():
        r = spearman_pair(cand, v)
        print(f"   spearman vs {k:24s} = {r:.4f}" if r is not None else f"   spearman vs {k}: n/a")
