"""Screener revalidation of the ONLY active library factor: usdcny_beta_60.
Caps data at visible_through = 2026-07-29 (no lookahead on decision day 07-30)."""
import sys, json, base64, zlib, io
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import ASSETS, load_closes, load_index, fwd_returns, ic_series, turnover_rank, coverage

END = pd.Timestamp("2026-07-29")
close, vol, open_, high, low = load_closes(end_date=END)
usdcny = load_index("USDCNY")

def _m(c, m, key):
    return m[key].reindex(c.index).ffill()

def usdcny_beta_60(c, v, o, h, l, m, win=60):
    ms = _m(c, m, "USDCNY")
    mr = ms.pct_change()
    r = c.pct_change()
    cov = r.rolling(win).cov(mr)
    var = mr.rolling(win).var()
    return cov / var

# build panel like factor_validation_lib.factor_panel
def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    out = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        c = close[a].reindex(idx); v = None if vol is None else vol[a].reindex(idx)
        o = None if open_ is None else open_[a].reindex(idx)
        h = None if high is None else high[a].reindex(idx)
        l = None if low is None else low[a].reindex(idx)
        try:
            s = fn(c, v, o, h, l, macro, **params)
            out[a] = pd.Series(s.values, index=idx).reindex(close.index)
        except Exception:
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)

macro = {"USDCNY": usdcny}
panel = factor_panel(usdcny_beta_60, close, vol, open_, high, low, macro)

# ---- compare with persisted artifact ----
art = json.load(open("factors/usdcny_beta_60.json"))["validation"]["signal_artifact"]["data"]
persist = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art)).decode()), index_col=0, parse_dates=True)
persist.index = pd.DatetimeIndex(persist.index)
persist = persist[persist.index <= END]
common = panel.join(persist, lsuffix="_r", rsuffix="_p").dropna()
if len(common):
    r = panel.reindex(common.index); p = persist.reindex(common.index)
    mask = r.notna() & p.notna()
    diff = (r[mask] - p[mask]).abs()
    print(f"persisted panel overlap: dates {len(common)}, non-nan pairs {int(mask.sum().sum())}")
    print(f"max abs diff vs persisted: {float(diff.max().max()) if diff.size else 'n/a'}")
print("my panel shape:", panel.shape, "non-nan:", int(panel.notna().sum().sum()))

cov_ad, cov8 = coverage(panel)
print(f"coverage asset-days={cov_ad:.4f} dates_ge8={cov8:.4f}")
to = turnover_rank(panel, lag=10)
print(f"turnover_10d_rank={to:.4f}")

# ---- IC on windows ----
fr = fwd_returns(close, 10)
ic_full = ic_series(panel, fr)
print(f"\nIC full: mean={ic_full.mean():+.4f} icir={ic_full.mean()/ic_full.std():+.4f} hit={(ic_full>0).mean():.3f} n={len(ic_full)}")
for label, nd in [("1m",21),("3m",63),("6m",126),("1y",252)]:
    sub = ic_full.iloc[-nd:] if len(ic_full) >= nd else ic_full
    if len(sub) >= 5:
        print(f"IC last {label}: mean={sub.mean():+.4f} icir={sub.mean()/sub.std():+.4f} hit={(sub>0).mean():.3f} n={len(sub)}")
    else:
        print(f"IC last {label}: n={len(sub)} too few")

# ---- recent coverage ----
for label, nd in [("3m",63),("6m",126)]:
    sub = panel.iloc[-nd:]
    c8 = float((sub.notna().sum(axis=1) >= 8).mean())
    print(f"recent {label} coverage: asset-days={float(sub.notna().sum().sum())/(sub.shape[0]*sub.shape[1]):.3f} dates_ge8={c8:.3f}")

# ---- regime snapshot ----
print("\n=== regime snapshot (through 07-29) ===")
for a in ASSETS:
    c = close[a].dropna()
    if len(c) < 30: continue
    r1 = c.iloc[-1]/c.iloc[-22]-1 if len(c)>22 else np.nan
    r3 = c.iloc[-1]/c.iloc[-64]-1 if len(c)>64 else np.nan
    vol21 = c.pct_change().tail(21).std()*np.sqrt(252)
    print(f"{a:10s} px={c.iloc[-1]:10.2f} 1M={r1*100:7.2f}% 3M={r3*100:7.2f}% vol21={vol21*100:5.1f}%")
for mname in ["DXY","USDCNY","VIX","USDJPY","EURUSD"]:
    s = load_index(mname).dropna()
    if len(s) < 30: continue
    r1 = s.iloc[-1]/s.iloc[-22]-1 if len(s)>22 else np.nan
    r3 = s.iloc[-1]/s.iloc[-64]-1 if len(s)>64 else np.nan
    print(f"{mname:10s} px={s.iloc[-1]:10.2f} 1M={r1*100:7.2f}% 3M={r3*100:7.2f}%")
