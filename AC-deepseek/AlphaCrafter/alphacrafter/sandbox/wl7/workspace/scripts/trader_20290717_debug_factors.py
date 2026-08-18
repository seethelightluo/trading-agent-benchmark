"""Debug: reproduce strategy factor computation at current date (2029-07-17)."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
print("assets:", assets)
print("n assets:", len(assets))

MIN_ROWS = 61


def _stock(sym, days=200):
    try:
        return get_stock_daily_data(sym, days=days)
    except Exception as e:
        print("  stock err", sym, e)
        return None


def _index(sym, days=200):
    try:
        return get_index_daily_data(sym, days=days)
    except Exception as e:
        print("  index err", sym, e)
        return None


def _series(df, name=None):
    if df is None or "close" not in df or len(df) < MIN_ROWS:
        return None
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    return s.rename(name) if name else s


frames = {a: _stock(a) for a in assets}
series = {a: _series(f) for a, f in frames.items()}
usable = {a: s.pct_change().rename(a) for a, s in series.items() if s is not None}
print("usable count:", len(usable))
if len(usable) < 8:
    print("FALLBACK: usable < 8")
R = pd.concat(usable, axis=1, join="inner").dropna().tail(150)
print("R shape:", R.shape, "last date:", R.index[-1])
if len(R) < MIN_ROWS:
    print("FALLBACK: R < MIN_ROWS")

cp = (1.0 + R).cumprod()
mkt = R.mean(axis=1)

# 1) rel_mom
mom = cp.shift(5) / cp.shift(25) - 1.0
rel_mom = mom.sub(mom.median(axis=1), axis=0)
print("rel_mom last:", rel_mom.iloc[-1].round(3).to_dict())

# 2) beta_ew
mvar = mkt.rolling(60).var()
beta_ew = R.rolling(60).cov(mkt).div(mvar, axis=0)
print("beta_ew last:", beta_ew.iloc[-1].round(3).to_dict())

# 3) downside_vol_ratio
neg = R.clip(upper=0.0)
semi = (neg ** 2).rolling(20).mean() ** 0.5
tot = R.rolling(20).std()
dvr = -(semi / tot)
print("dvr last:", dvr.iloc[-1].round(3).to_dict())

# 4) max_ret
mx = R.rolling(20).max()
print("max_ret last:", mx.iloc[-1].round(3).to_dict())

# 5) dxy
dxy_cond = None
dfx = _index("DXY")
if dfx is not None and len(dfx) >= MIN_ROWS:
    dc = pd.Series(dfx["close"].astype(float), index=pd.to_datetime(dfx["date"]))
    dxy_ret = dc.pct_change().reindex(R.index)
    dxy_20 = (dc / dc.shift(20) - 1.0).reindex(R.index)
    if dxy_ret.notna().sum() >= 40 and dxy_20.notna().sum() >= 40:
        dvar = dxy_ret.rolling(60).var()
        bfx = R.rolling(60).cov(dxy_ret).div(dvar, axis=0)
        dxy_cond = -bfx * dxy_20
        print("dxy_cond last:", dxy_cond.iloc[-1].round(3).to_dict())
    else:
        print("dxy skip: insufficient")
else:
    print("dxy skip: dfx None or short")

# 6) corr_ew
corr_parts = []
for a in R.columns:
    others = [R[a].rolling(60).corr(R[b]) for b in R.columns if b != a]
    corr_parts.append(pd.concat(others, axis=1).mean(axis=1).rename(a))
corr_ew = pd.concat(corr_parts, axis=1)
print("corr_ew last:", corr_ew.iloc[-1].round(3).to_dict())

# 7) kurt
kurt = R.shift(5).rolling(20).kurt()
print("kurt last:", kurt.iloc[-1].round(3).to_dict())

# ensemble
try:
    ens = json.loads((Path("factor_ensemble.json")).read_text())
    sel = [(str(it["factor_id"]), float(it["weight"]), int(it.get("direction", 1)))
           for it in ens.get("selected_factors", []) if isinstance(it, dict) and it.get("factor_id")]
    print("sel:", sel)
except Exception as e:
    print("ens err:", e)
    sel = []
