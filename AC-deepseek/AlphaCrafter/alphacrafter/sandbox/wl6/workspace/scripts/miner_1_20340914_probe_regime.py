"""miner_1 2034-09-14 regime probe + frozen asset snapshot via visible_through."""
import json
import numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("fl", "scripts/miner_1_20281102_fastlib.py")
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)

px = fl.load_close_panel()
obs = fl.load_macro_panel()
ret = px.pct_change()

frozen = [s for s in px.columns if (len(ret[s].dropna()) and ret[s].dropna().iloc[-250:].abs().max() < 1e-12) or px[s].nunique() <= 1]
active = [s for s in px.columns if s not in frozen]
print(f"rows={len(px)} last={px.index[-1].date()} frozen={frozen} active={len(active)}")

print("=== recent returns ===")
for h in (5, 20, 60, 250):
    if len(px) > h:
        r = (px.iloc[-1] / px.iloc[-1-h] - 1.0)
        print(f"{h}d:", r.sort_values(ascending=False).round(3).to_dict())

print("\n=== macro observables ===")
for o in obs.columns:
    v = obs[o].dropna()
    print(f"{o}: last={v.iloc[-1]:.2f} 5d={v.iloc[-1]/v.iloc[-6]-1:+.3f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.3f} 60d={v.iloc[-1]/v.iloc[-61]-1:+.3f}")

spx = px['SPX'].dropna()
print("\n=== SPX trend ===")
for w in (20, 60, 120, 200):
    m = spx.rolling(w).mean().iloc[-1]
    mx = spx.rolling(w).mean().iloc[-1-w] if len(spx) > w else np.nan
    print(f"ma{w}={m:.0f} last={spx.iloc[-1]:.0f} above={spx.iloc[-1]>m} slope={ (m/mx-1)*100 if not np.isnan(mx) else np.nan:+.2f}%")

vol20_assets = ret[active].std(axis=1).tail(20).mean()
print(f"\ncross-sectional mean daily dispersion 20d={vol20_assets:.4f}")