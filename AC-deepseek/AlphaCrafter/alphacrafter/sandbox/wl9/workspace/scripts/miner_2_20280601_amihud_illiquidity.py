"""
miner_2 2028-06-01: Amihud illiquidity z-score factor.
Idea: in a 15-asset cross-asset universe, assets whose price impact per unit
volume (Amihud = |ret|/volume) is unusually high relative to their own history
may mean-revert (recent illiquidity stress) or earn a liquidity premium.
Construction: z = (amihud - rolling_mean)/rolling_std, amihud = |pct_change|/volume
per asset, then cross-sectional rank. Direction determined empirically
(report both +z and -z IC). Validation on full available history 2020-01..2028-05.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MIN_ASSETS = 8
ADM_IC, ADM_ICIR = 0.0070, 0.0840

# ---- Load data ----
close_dfs, vol_dfs = {}, {}
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=2500)
    if df is None or len(df) < 300:
        print(f"{sym}: insufficient data, skip")
        continue
    df = df.set_index("date").sort_index()
    close_dfs[sym] = df["close"]
    v = df["volume"].replace(0, np.nan)
    vol_dfs[sym] = v

close_df = pd.DataFrame(close_dfs).sort_index()
vol_df = pd.DataFrame(vol_dfs).sort_index()
print(f"close panel: {close_df.shape}, date range {close_df.index.min()} .. {close_df.index.max()}")

ret_df = close_df.pct_change()
amihud = (ret_df.abs() / vol_df).replace([np.inf, -np.inf], np.nan)

def build_factor(z_window=60, q=3):
    # truncate extreme amihud before z-scoring
    am = amihud.clip(lower=amihud.quantile(0.01), upper=amihud.quantile(0.99), axis=0)
    mu = am.rolling(z_window, min_periods=30).mean()
    sd = am.rolling(z_window, min_periods=30).std()
    z = (am - mu) / sd.replace(0, np.nan)
    # median-filter rank stability: winsorize the z cross-section each date
    zz = z.rank(axis=1, pct=True) * 2.0 - 1.0
    return zz

def ic_stats(factor_df, horizon=10, label=""):
    fwd = close_df.pct_change(horizon).shift(-horizon)
    common = factor_df.index.intersection(fwd.index)
    ics = []
    ndates_ge8 = 0
    for dt in common:
        f = factor_df.loc[dt]
        r = fwd.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= MIN_ASSETS:
            ndates_ge8 += 1
            ff, rr = f[m].rank(), r[m].rank()
            if len(ff) >= 3:
                rho = np.corrcoef(ff, rr)[0, 1]
                if not np.isnan(rho):
                    ics.append(rho)
    ics = np.array(ics)
    if len(ics) == 0:
        print(f"{label}: no IC obs"); return None
    mean_ic, std_ic = ics.mean(), ics.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = (ics > 0).mean()
    # turnover: rank autocorr at 10d
    f10 = factor_df.resample("10D").last()
    ta = f10.corrwith(f10.shift(1)).mean()
    print(f"{label}: IC={mean_ic:.5f} ICIR={icir:.5f} hit={hit:.4f} n_obs={len(ics)} "
          f"dates_ge8={ndates_ge8} turnover10d_rank_autocorr={ta:.4f}")
    return {"ic": mean_ic, "icir": icir, "hit": hit, "n_obs": len(ics), "ndates_ge8": ndates_ge8, "turnover_auto": ta}

print("\n=== Amihud z-score, z_window=60, horizon=10 ===")
z = build_factor(60)
s = ic_stats(z, 10, "z(amihud,60) +dir")
sneg = ic_stats(-z, 10, "z(amihud,60) -dir(negated)")
if s is not None:
    print(f"Gate +dir: |IC|={abs(s['ic']):.5f} >= {ADM_IC} -> {abs(s['ic']):.5f}>= {ADM_IC:.4f} "
          f"|ICIR|={abs(s['icir']):.5f} >= {ADM_ICIR} -> {abs(s['icir']):.5f} >= {ADM_ICIR:.4f}")
    print(f"Gate -dir: |IC|={abs(sneg['ic']):.5f} |ICIR|={abs(sneg['icir']):.5f}")

print("\n=== Decay by horizon (z60) ===")
for h in [1, 2, 3, 5, 10, 20]:
    ic_stats(z, h, f"z60 h={h}")

print("\n=== Window sensitivity (horizon=10) ===")
for w in [20, 40, 60, 90]:
    ic_stats(build_factor(w), 10, f"z_window={w}")

# Sub-period stability
print("\n=== Sub-period split (z60, h10) ===")
fwd = close_df.pct_change(10).shift(-10)
common = z.index.intersection(fwd.index)
split = common[len(common)//2]
for name, mask in [("early", common < split), ("late", common >= split)]:
    ics = []
    for dt in common[mask]:
        f = z.loc[dt]; r = fwd.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= MIN_ASSETS:
            ff, rr = f[m].rank(), r[m].rank()
            if len(ff) >= 3:
                rho = np.corrcoef(ff, rr)[0, 1]
                if not np.isnan(rho):
                    ics.append(rho)
    ics = np.array(ics)
    if len(ics):
        print(f"  {name}: IC={ics.mean():.5f} ICIR={ics.mean()/ics.std():.5f} n={len(ics)} (split {split.date()})")