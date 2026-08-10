"""Diagnose 1396-column macro-beta candidates in screen_broad.py."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
print("idx dates:", len(idx), idx.min().date(), idx.max().date(), "dup:", idx.duplicated().sum())

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
print("RET shape:", RET.shape, "cols:", len(RET.columns))

macro_dir = "../persistent/index_data"
def load_macro(name):
    d = pd.read_csv(os.path.join(macro_dir, f"{name}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= pd.Timestamp("2026-07-15")].set_index("date")["close"].astype(float)
    return d.reindex(idx).ffill()
DXY = load_macro("DXY"); VIX = load_macro("VIX"); USDCNY = load_macro("USDCNY")
dxy_r = DXY.pct_change(); vix_c = VIX.diff(); usdcny_r = USDCNY.pct_change()
for nm, x in [("dxy_r", dxy_r), ("vix_c", vix_c), ("usdcny_r", usdcny_r)]:
    print(nm, type(x).__name__, x.shape, "dup idx:", x.index.duplicated().sum(), "name:", x.name)

def roll_beta(y_panel, x, win):
    out = {}
    for s in y_panel.columns:
        y = y_panel[s]
        df = pd.concat([y, x], axis=1).dropna()
        print(f"  {s}: concat cols={df.shape[1]} rows={len(df)} dup_idx={df.index.duplicated().sum()} "
              f"coltypes={[type(c).__name__ for c in df.columns]}")
        cov = df.iloc[:, 0].rolling(win).cov(df.iloc[:, 1])
        var = df.iloc[:, 1].rolling(win).var()
        print(f"     cov type={type(cov).__name__} shape={getattr(cov,'shape',None)} "
              f"var type={type(var).__name__} shape={getattr(var,'shape',None)}")
        out[s] = (cov / var).reindex(idx)
        print(f"     out[{s}] type={type(out[s]).__name__} shape={getattr(out[s],'shape',None)}")
    return pd.DataFrame(out)

b = roll_beta(RET, dxy_r, 60)
print("roll_beta result:", b.shape, "cols:", b.columns.tolist()[:5])

# Now test the 'usd_tilt' combination which multiplies DataFrame x Series
dxy_trend = np.sign(dxy_r.rolling(60).mean())
tilt = -b * dxy_trend
print("usd_tilt:", tilt.shape, type(tilt).__name__)
