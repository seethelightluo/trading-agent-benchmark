"""Preview regime metrics at the 2031-10-16 decision (block start)."""
import json, math
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY","VIX","USDCNY","USDJPY","EURUSD"}
EQ_ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX"]

def get_df(symbol, days=260):
    try:
        if symbol in OBS_ONLY: return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None

def series(df, col="close"):
    if df is None or col not in df or len(df) < 40: return None
    s = df[col].astype(float)
    try: s.index = pd.to_datetime(df["date"])
    except Exception: s.index = pd.RangeIndex(len(s))
    return s

assets = list(get_account_dict()["watch_list"])
frames = {a: get_df(a) for a in assets}
close = {a: series(frames[a]) for a in assets}

def detect_frozen(close, lookback=120):
    out = set()
    for a, c in close.items():
        if c is None: continue
        q = c.dropna().tail(lookback)
        if len(q) >= 20 and q.nunique() <= 2: out.add(a)
    return out

frozen = detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live:", live)

ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
print("panel rows:", len(panel), "last date:", panel.index[-1])

lp = panel[live] if live else panel
market = lp.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(lp.tail(20).std().mean())
vol_med = float(lp.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
def_floor = 0.18 if risk_off else (0.11 if risk_on else 0.13)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)
print(f"mkt20={mkt20*100:.2f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% vol_med={vol_med*100:.2f}% vol_ratio={vol20/max(vol_med,1e-6):.3f}")
print(f"risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread}")

vix = series(get_df("VIX"))
vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in EQ_ASSETS if a in live]
eq_ret21 = float(np.mean([close[a].iloc[-1]/close[a].iloc[-22]-1.0 for a in eq_live])) if eq_live else 0.0
stress = risk_off and ((vix_level is not None and vix_level >= 30.0) or eq_ret21 < -0.05)
print(f"VIX={vix_level:.2f} eq21={eq_ret21*100:.2f}% stress(current logic)={stress}")

# what a VIX-only trigger would say
stress_vixonly = (vix_level is not None and vix_level >= 30.0)
print(f"stress(VIX-only trigger)={stress_vixonly} -> EQ_CAP 0.40, ETH_CAP 0.06")
