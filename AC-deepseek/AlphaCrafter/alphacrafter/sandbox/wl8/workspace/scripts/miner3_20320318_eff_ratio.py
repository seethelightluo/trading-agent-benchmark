"""
Miner 3: Efficiency Ratio (Kaufman) Factor Validation (2032-03-18)
Factor: eff_ratio_20 = |close - close_{20}| / sum(|close_i - close_{i-1}|, i=1..20)
Measures directional efficiency of price movement.
Higher = trending/high signal, Lower = noisy/choppy.
In high-vol divergent regimes, efficient trending assets may continue,
while noisy assets are more likely to mean-revert.
"""
import numpy as np
import pandas as pd

CURRENT_DATE = pd.Timestamp("2032-03-18")
ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS = 8

print(f"Current date: {CURRENT_DATE.date()}")
print(f"Assets: {ASSETS}")
print(f"IC Gate: |IC| >= {IC_GATE}, |ICIR| >= {ICIR_GATE}")

# Load close data
closes = {}
for a in ASSETS:
    df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= CURRENT_DATE].set_index("date").sort_index()
    closes[a] = df["close"].astype(float)

close_df = pd.DataFrame(closes)
print(f"Close data: {close_df.shape}, {close_df.index[0].date()} -> {close_df.index[-1].date()}")

# --- Factor 1: EFF_RATIO_20 ---
print("\n=== Factor 1: EFF_RATIO_20 ===")
factor_raw = {}
for a in ASSETS:
    c = closes[a].dropna().copy()
    window = 20
    # Directional movement: |close_t - close_{t-20}|
    direction = (c - c.shift(window)).abs()
    # Total path length: sum of absolute daily changes over window
    path = c.diff().abs().rolling(window, min_periods=10).sum()
    er = direction / path
    factor_raw[a] = er.reindex(close_df.index)

f1 = pd.DataFrame(factor_raw)

# Coverage
n_total = f1.notna().sum().sum()
n_cells = f1.shape[0] * f1.shape[1]
cov_ad = n_total / n_cells
cov_ge8 = (f1.notna().sum(axis=1) >= MIN_ASSETS).mean()
print(f"Coverage (asset-days): {cov_ad:.4f}")
print(f"Coverage (dates >= 8 assets): {cov_ge8:.4f}")

# IC at various horizons
horizons = [1, 2, 3, 5, 10, 20]
results = {}
for h in horizons:
    fwd_ret = close_df.shift(-h) / close_df - 1.0
    ics = []
    for dt in f1.index:
        x = f1.loc[dt]
        y = fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            ics.append(x[m].rank().corr(y[m].rank()))
    if len(ics) > 2:
        ic_arr = np.array(ics)
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr, ddof=1) if len(ic_arr) > 2 else 1.0
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        hit = np.mean((ic_arr > 0) if ic_mean >= 0 else (ic_arr < 0))
        results[h] = {"ic": ic_mean, "icir": icir, "hit": hit, "n": len(ic_arr)}
        print(f"  H={h:2d}: IC={ic_mean:.6f}, ICIR={icir:.6f}, hit={hit:.4f}, n={len(ic_arr)}")
    else:
        results[h] = {"ic": np.nan, "icir": np.nan, "hit": np.nan, "n": len(ics)}
        print(f"  H={h:2d}: insufficient data (n={len(ics)})")

# Primary horizon = 10
h_primary = 10
ic10 = results[10]["ic"]
icir10 = results[10]["icir"]
ic10_hit = results[10]["hit"]
n10 = results[10]["n"]

gate_pass = abs(ic10) >= IC_GATE and abs(icir10) >= ICIR_GATE
print(f"\n  Admission H=10: IC={ic10:.6f}, ICIR={icir10:.6f}")
print(f"  Gate: |IC|>={IC_GATE} ({abs(ic10) >= IC_GATE}), |ICIR|>={ICIR_GATE} ({abs(icir10) >= ICIR_GATE})")
print(f"  PASS={gate_pass}")

# Turnover
ranks = f1.rank(axis=1)
turn = ranks.diff(10).abs().mean(axis=1).dropna().mean()
print(f"  Turnover (10d rank change): {turn:.4f}")

print(f"\n=== Summary: EFF_RATIO_20 ===")
print(f"  H=10 IC={ic10:.6f}, ICIR={icir10:.6f}, Hit={ic10_hit:.4f}, n={n10}")
print(f"  Coverage_AD={cov_ad:.4f}, GE8={cov_ge8:.4f}, Turnover={turn:.4f}")
print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

if gate_pass:
    print(f"\n  *** CANDIDATE PASSES GATE ***")
    
print("\nDecay analysis:")
for h in horizons:
    r = results[h]
    print(f"  H={h:2d}: IC={r['ic']:.6f}, ICIR={r['icir']:.6f}")