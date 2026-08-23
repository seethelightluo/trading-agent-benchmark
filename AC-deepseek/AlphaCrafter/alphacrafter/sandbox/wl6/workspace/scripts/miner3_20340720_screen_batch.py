"""miner_3 screen batch (2034-07-20), VIS 2034-07-19. Fresh candidate factors for the
15-instrument cross-asset universe plus macro observation series.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Report recent (last 2y) instability too. Prints n dates/instruments used.
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-07-19"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill()
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1])

def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES")
        return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-01-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

# A. Skewness 20d (negative skew premium)
skew20 = ret.rolling(20).skew()
cands["skew_20_pos"] = skew20  # positive skew? test

# B. Vol-of-vol change (regime shift): stdev of 20d vol over 60d
rv20 = ret.rolling(20).std()
cands["vol_of_vol_chg_20"] = rv20.rolling(60).std()

# C. Cross-section: 60d realized vol vs SPX vol (relative vol tilt)
spx_rv = rv20["SPX"]
cands["rel_vol_spx_60"] = rv20 / spx_rv.replace(0, np.nan)

# D. Momentum 30d skip10
cands["mom_30_skip10"] = px.shift(10)/px.shift(40) - 1.0

# E. Long-term mean reversion: 200d distance from SMA100
sma100 = px.rolling(100).mean()
cands["ma100_dist_200"] = (px/sma100 - 1.0)

# F. EURUSD beta (weak-dollar carry)
eur = load_macro("EURUSD", VIS).reindex(px.index).ffill()
emom = (eur/eur.shift(20)-1).reindex(px.index).ffill()
ebeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    ebeta[a] = ret[a].rolling(60).corr(emom)
cands["eur_beta_60"] = ebeta

# G. Range contraction (volatility squeeze -> breakout): 20d range / 60d range
rng20 = px.rolling(20).max()/px.rolling(20).min() - 1.0
rng60 = px.rolling(60).max()/px.rolling(60).min() - 1.0
cands["range_ratio_20x60"] = rng20/rng60.replace(0, np.nan)

# H. Downside capture: mean negative-return day size normalized
down = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    down[a] = ret[a].rolling(60).apply(lambda x: np.mean(x[x<0]) if (x<0).any() else np.nan, raw=True)
cands["downside_neg"] = -down

for name, f in cands.items():
    evalc(f, name)