"""
Revalidate all VIX-related factors as of 2035-12-06.
Also compute candidate VIX Acceleration factor.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
from scipy.stats import spearmanr

acct = get_account_dict()
watchlist = acct.get('watch_list', [])
print(f"Watchlist: {watchlist}")
print(f"Number of assets: {len(watchlist)}")

vix = get_index_daily_data("VIX", 500)
vix_close = vix.set_index('date')['close']
print(f"VIX: {len(vix)} rows, end={vix['date'].iloc[-1].strftime('%Y-%m-%d')}, cur={vix_close.iloc[-1]:.2f}")

asset_data = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    if df is not None and len(df) >= 60:
        asset_data[sym] = df

price_panel = {}
for sym, df in asset_data.items():
    price_panel[sym] = df[['date', 'close']].set_index('date')['close']
combined = pd.DataFrame(price_panel).dropna(how='all')
print(f"Price panel: {combined.shape}")

fwd_10 = combined.shift(-10) / combined - 1
asset_rets = combined.pct_change()
common_ix = asset_rets.index.intersection(vix_close.index)
vix_aligned = vix_close.loc[common_ix]
vix_ret = vix_aligned.pct_change()
asset_rets_a = asset_rets.loc[common_ix]
fwd_a = fwd_10.loc[common_ix]
n = len(common_ix)
print(f"Common dates: {n}")

window = 60

# Helper to compute IC
def compute_ic(signal_df, fwd_df, min_assets=8):
    sig_s = signal_df.stack().dropna().reset_index()
    sig_s.columns = ['date','asset','signal']
    merged = sig_s.merge(fwd_df, on=['date','asset'], how='inner')
    if len(merged) == 0:
        return 0, 0, 0, 0
    ics = []
    for dt, grp in merged.groupby('date'):
        if len(grp) >= min_assets:
            ic, _ = spearmanr(grp['signal'], grp['fwd_ret'])
            ics.append(ic)
    if len(ics) == 0:
        return 0, 0, 0, 0
    ic_m = np.mean(ics)
    ic_s = np.std(ics)
    icir = ic_m / ic_s if ic_s > 0 else 0
    return ic_m, icir, len(ics), len(merged)

# Prepare flat forward returns
fwd_s = fwd_a.stack().dropna().reset_index()
fwd_s.columns = ['date','asset','fwd_ret']

# ===== 1. beta_VIX_60 =====
print("\n========= beta_VIX_60 =========")
beta_sig = pd.DataFrame(index=common_ix, columns=combined.columns, dtype=float)
for t in range(window, n):
    idx = common_ix[t]
    for c in combined.columns:
        r_s = asset_rets_a[c].iloc[t-window+1:t+1].dropna()
        v_s = vix_ret.iloc[t-window+1:t+1]
        ci = r_s.index.intersection(v_s.index)
        if len(ci) >= 20:
            rv = r_s.loc[ci].values
            vv = v_s.loc[ci].values
            cov = np.cov(rv, vv)[0,1]
            varv = np.var(vv)
            if varv > 1e-10:
                beta_sig.loc[idx, c] = cov / varv
ic_m, icir, nd, nobs = compute_ic(beta_sig, fwd_s)
print(f"IC={ic_m:.6f}, ICIR={icir:.6f}, n_dates={nd}, nobs={nobs}")
print(f"PASS threshold (absIC>=0.007, absICIR>=0.084): {abs(ic_m)>=0.007 and abs(icir)>=0.084}")

# ===== 2. vix_beta_cond_60x20 =====
print("\n========= vix_beta_cond_60x20 =========")
cond_sig = pd.DataFrame(index=common_ix, columns=combined.columns, dtype=float)
for t in range(window, n):
    idx = common_ix[t]
    for c in combined.columns:
        r_s = asset_rets_a[c].iloc[t-window+1:t+1].dropna()
        v_s = vix_ret.iloc[t-window+1:t+1]
        ci = r_s.index.intersection(v_s.index)
        if len(ci) >= 20:
            rv = r_s.loc[ci].values
            vv = v_s.loc[ci].values
            cov = np.cov(rv, vv)[0,1]
            varv = np.var(vv)
            if varv > 1e-10:
                beta_val = cov/varv
                vixmove = vix_aligned.loc[idx]/vix_aligned.shift(20).loc[idx] - 1
                cond_sig.loc[idx, c] = -beta_val * vixmove
ic_m, icir, nd, nobs = compute_ic(cond_sig, fwd_s)
print(f"IC={ic_m:.6f}, ICIR={icir:.6f}, n_dates={nd}, nobs={nobs}")
print(f"PASS: {abs(ic_m)>=0.007 and abs(icir)>=0.084}")

# ===== 3. vix_roc_20d =====
print("\n========= vix_roc_20d =========")
safe_havens = ['XAU','US10Y','CN10Y']
roc20 = vix_aligned / vix_aligned.shift(20) - 1
roc_sig = pd.DataFrame(index=common_ix, columns=combined.columns, dtype=float)
for idx in common_ix[max(window,20):]:
    for c in combined.columns:
        vix_roc_val = roc20.loc[idx]
        if c in safe_havens:
            roc_sig.loc[idx,c] = vix_roc_val
        else:
            roc_sig.loc[idx,c] = -vix_roc_val
ic_m, icir, nd, nobs = compute_ic(roc_sig, fwd_s)
print(f"IC={ic_m:.6f}, ICIR={icir:.6f}, n_dates={nd}, nobs={nobs}")
print(f"PASS: {abs(ic_m)>=0.007 and abs(icir)>=0.084}")

# ===== 4. CANDIDATE: VIXX Acceleration Factor =====
print("\n========= CANDIDATE: vix_accel_20_40 =========")
vix_roc20_series = vix_aligned / vix_aligned.shift(20) - 1
vix_accel = vix_roc20_series - vix_roc20_series.shift(20)

accel_sig = pd.DataFrame(index=common_ix, columns=combined.columns, dtype=float)
for idx in common_ix[max(window,40):]:
    accel_val = vix_accel.loc[idx]
    for c in combined.columns:
        if c in safe_havens:
            accel_sig.loc[idx,c] = accel_val
        else:
            accel_sig.loc[idx,c] = -accel_val
ic_m, icir, nd, nobs = compute_ic(accel_sig, fwd_s)
print(f"IC={ic_m:.6f}, ICIR={icir:.6f}, n_dates={nd}, nobs={nobs}")
print(f"PASS: {abs(ic_m)>=0.007 and abs(icir)>=0.084}")

# ===== 5. CANDIDATE: Volatility Regime Level (vix_level_z_60) =====
print("\n========= CANDIDATE: vix_level_z_60 =========")
vix_mean_60 = vix_aligned.rolling(60).mean()
vix_std_60 = vix_aligned.