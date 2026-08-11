"""Replicate regime posture at 2027-05-20 block start (data visible then)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def get_df(sym):
    try:
        return get_stock_daily_data(sym, days=200) if sym not in OBS else get_index_daily_data(sym, days=200)
    except Exception:
        return None

rets = {}
for a in WATCH:
    df = get_df(a)
    if df is None:
        continue
    s = df["close"].astype(float)
    s.index = pd.to_datetime(df["date"])
    rets[a] = s.pct_change()
panel = pd.concat(rets, axis=1).dropna()
panel = panel[panel.index <= "2027-05-20"]

market = panel.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(panel.tail(20).std().mean())
vol_med = float(panel.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.03) or (vol20 > 1.3 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.02
def_floor = 0.15 if risk_off else (0.10 if risk_on else 0.12)
spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)
print(f"mkt20={mkt20:.5f} mdd20={mdd:.5f} vol20={vol20:.5f} vol_med120={vol_med:.5f}")
print(f"risk_off={risk_off} risk_on={risk_on} def_floor={def_floor} spread={spread}")
