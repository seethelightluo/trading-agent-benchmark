"""
Factor: vol_stability_ratio_20x60
Idea: Ratio of short-term (20d) volatility to long-term (60d) volatility.
When short-term vol is much higher than long-term vol, the asset is experiencing
volatility expansion (instability). When short-term vol is lower, it's in a 
compression phase. This is orthogonal to USDCNY beta signals.

Hypothesis: Assets with vol compression (ratio < 1) tend to outperform in the 
near future as they are building up for a breakout, while assets with vol expansion
(ratio > 1) are in disorderly regimes and may underperform.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
CURRENT_DATE = pd.Timestamp("2032-05-13")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS = 8

def load_closes():
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= CURRENT_DATE].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
    return pd.DataFrame(closes)

def factor_fn(close):
    """Volatility stability ratio: 20d vol / 60d vol"""
    ret = close.pct_change()
    vol_20 = ret.rolling(20, min_periods=10).std()
    vol_60 = ret.rolling(60, min_periods=30).std()
    ratio = vol_20 / vol_60
    # Clip extreme values
    ratio = ratio.clip(0.1, 5.0)
    # Negative: high ratio (vol expansion) = bad, low ratio (vol compression) = good
    # So we negate: signal = -ratio (or 1 - ratio)
    signal = 1.0 - ratio  # High when vol compressing, low when expanding
    return signal

def fwd_returns(close, horizon):
    fr = close.shift(-horizon) / close - 1.0
    return fr

def ic_series(factor, ret):
    ics = []
    dates = []
    for dt in factor.index:
        x = factor.loc[dt]
        y = ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

print("Loading data...")
close = load_closes()
print(f"Close data: {close.shape[0]} dates x {close.shape[1]} assets")
print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

print("\nComputing factor...")
signal = factor_fn(close)
print(f"Signal shape: {signal.shape}")
print(f"Signal coverage: {signal.notna().sum().sum() / (signal.shape[0]*signal.shape[1]):.4f}")

# Test horizons
horizons = [1, 2, 3, 5, 10, 20]
results = {}
for h in horizons:
    fr = fwd_returns(close, h)
    ic = ic_series(signal, fr)
    ic_mean = float(ic.mean())
    ic_std = float(ic.std()) if len(ic) > 2 else np.nan
    icir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else 0.0
    hit = float((ic > 0).mean()) if ic_mean >= 0 else float((ic < 0).mean())
    results[h] = {
        "ic": round(ic_mean, 6),
        "icir": round(icir, 6) if np.isfinite(icir) else 0.0,
        "hit_ratio": round(hit, 4),
        "n_dates": len(ic),
    }
    print(f"  H={h:2d}: IC={results[h]['ic']:.6f}, ICIR={results[h]['icir']:.6f}, "
          f"Hit={results[h]['hit_ratio']:.4f}, n={results[h]['n_dates']}")

# Check admission horizon (10)
adm = results[10]
print(f"\nAdmission (H=10): IC={adm['ic']:.6f}, ICIR={adm['icir']:.6f}")
print(f"Gate: |IC|>={IC_GATE} and |ICIR|>={ICIR_GATE}")
passes = abs(adm['ic']) >= IC_GATE and abs(adm['icir']) >= ICIR_GATE
print(f"Result: {'PASS' if passes else 'FAIL'}")

# Check factor coverage
cov = signal.notna().sum(axis=1)
ge8 = (cov >= MIN_ASSETS).mean()
print(f"\nCoverage: dates with >={MIN_ASSETS} assets: {ge8:.4f}")

# Check turnover
ranks = signal.rank(axis=1)
turn = ranks.diff(10).abs().mean(axis=1).mean()
print(f"Turnover (10d rank change): {turn:.4f}")