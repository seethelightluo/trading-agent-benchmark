"""miner_2 2027-06-17: probe WHY xau_beta_60 is valid only from ~2026."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_asset(symbol, days=3200):
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = get_stock_daily_data(symbol=symbol, days=days)
        except Exception:
            df = None
    if df is None or len(df) < 400:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


data = {a: load_asset(a) for a in WATCH}
closes = {a: d["close"].astype(float) for a, d in data.items()}
fclose = pd.DataFrame(closes).sort_index()
rets = fclose.pct_change()

# XAU return structure
rxau_full = rets["XAU"]
print("XAU ret: total", rxau_full.notna().sum(), "first valid", rxau_full.first_valid_index(),
      "last valid", rxau_full.last_valid_index())
print("XAU close first 3:\n", fclose["XAU"].head(3))
print("XAU close NaN count:", fclose["XAU"].isna().sum())
print("XAU ret NaN count:", rxau_full.isna().sum())

# manual vs rolling for SPX
r = closes["SPX"].pct_change()
print("\nSPX ret: total", r.notna().sum(), "first valid", r.first_valid_index())
# rolling var of XAU ret
vx = rxau_full.rolling(60).var()
print("XAU rolling var 60: valid", vx.notna().sum(), "first valid", vx.first_valid_index())
# rolling cov SPX vs XAU
cv = r.rolling(60).cov(rxau_full)
print("SPX-XAU rolling cov 60: valid", cv.notna().sum(), "first valid", cv.first_valid_index())
beta = cv / vx
print("xau_beta SPX: valid", beta.notna().sum(), "first valid", beta.first_valid_index())
print("xau_beta SPX last values:\n", beta.dropna().tail(3))

# check whether SPX and XAU returns share overlapping non-null values in early period
both = pd.concat([r, rxau_full], axis=1, keys=["spx", "xau"])
print("\nearly overlap 2020-01..2020-03:")
sub = both.loc["2020-01-01":"2020-03-31"]
print(sub.dropna().shape, "of", sub.shape)
print(sub.head(5))
