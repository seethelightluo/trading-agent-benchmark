"""Factor Miner 1 - Batch factor exploration (2032-03-04)
Current date: 2032-03-04 (visible through 2032-03-03)
Testing several novel factor ideas that haven't been tried before.

Factors to test:
1. dist_ma_20: Distance from 20-day moving average (close - sma_20) / sma_20
2. dist_ma_60: Distance from 60-day moving average
3. cross_sectional_vol_rank: Negative rank of 20-day volatility (favors low vol assets)
4. ma_cross_20x60: (sma_20 / sma_60 - 1) - moving average crossover
5. cvar_20: Conditional VaR - mean of returns below 5th percentile (negative = protective)
"""
import numpy as np
import pandas as pd
import sys, json, base64, zlib, io

sys.path.insert(0, '.')
from scripts.factor_validation_lib import (
    ASSETS, DATA_DIR, INDEX_DIR, IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE,
)

CURRENT_DATE = pd.Timestamp("2032-03-03")

# Monkey-patch module
import scripts.factor_validation_lib as fvl
fvl.CURRENT_DATE = CURRENT_DATE

def load_closes(end_date):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float); vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float); highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    return pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens), pd.DataFrame(highs), pd.DataFrame(lows)

def load_index(name):
    df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= CURRENT_DATE].set_index("date").sort_index()
    return df["close"].astype(float)

close, vol, open_, high, low = load_closes(CURRENT_DATE)
dxy = load_index("DXY"); vix = load_index("VIX"); usdcny = load_index("USDCNY")
macro = {"DXY": dxy, "VIX": vix, "USDCNY": usdcny}

# ========== FACTOR DEFINITIONS ==========

def factor_dist_ma_20(c, v, o, h, l, macro):
    """Distance from 20-day SMA: (close - sma_20) / sma_20. 
    Positive = above MA (momentum), negative = below MA (mean reversion opportunity)."""
    sma = c.rolling(20).mean()
    return (c / sma - 1.0)

def factor_dist_ma_60(c, v, o, h, l, macro):
    """Distance from 60-day SMA."""
    sma = c.rolling(60).mean()
    return (c / sma - 1.0)

def factor_ma_cross_20x60(c, v, o, h, l, macro):
    """MA crossover signal: sma_20 / sma_60 - 1. Positive = short-term trending above long-term."""
    sma20 = c.rolling(20).mean()
    sma60 = c.rolling(60).mean()
    return (sma20 / sma60 - 1.0)

def factor_vol_rank_neg(c, v, o, h, l, macro):
    """Cross-sectional negative rank of 20-day volatility.
    Low-vol assets get high scores. Defensive tilt."""
    vol20 = c.pct_change().rolling(20).std()
    return -vol20.rank(axis=1)

def factor_cvar_20_neg(c, v, o, h, l, macro):
    """Negative of 20-day CVaR (5% tail). CVaR = mean of returns below 5th percentile.
    Assets with less negative tail risk get higher scores."""
    r = c.pct_change()
    cvar = r.rolling(20).apply(
        lambda x: x[x < np.percentile(x.dropna(), 5) if len(x.dropna()) > 5 else 0].mean(),
        raw=False
    )
    return -cvar

# ========== RUN ALL ==========
candidates = [
    ("dist_ma_20", factor_dist_ma_20, {"lookback": 20}),
    ("dist_ma_60", factor_dist_ma_60, {"lookback": 60}),
    ("ma_cross_20x60", factor_ma_cross_20x60, {"short": 20, "long": 60}),
    ("vol_rank_neg_20", factor_vol_rank_neg, {"window": 20}),
    ("cvar_20_neg", factor_cvar_20_neg, {"window": 20}),
]

results = {}
for name, fn, params in candidates:
    print(f"\n{'='*60}")
    print(f"VALIDATING: {name}")
    print(f"{'='*60}")
    panel = fvl.factor_panel(fn, close, vol, open_, high, low, macro)
    
    # IC series at 10-day horizon
    fr = fvl.fwd_returns(close, 10)
    ic_ser = fvl.ic_series(panel, fr)
    
    ic = float(ic_ser.mean()) if len(ic_ser) else 0
    icir = float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 2 else 0
    hit = float((ic_ser > 0).mean()) if np.isfinite(ic) else 0
    if ic < 0:
        hit = float((ic_ser < 0).mean())
    
    # Decay
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fr_h = fvl.fwd_returns(close, h)
        ic_h = fvl.ic_series(panel, fr_h)
        decay[h] = float(ic_h.mean()) if len(ic_h) else np.nan
    
    # Coverage
    cov_ad = float(panel.notna().sum().su