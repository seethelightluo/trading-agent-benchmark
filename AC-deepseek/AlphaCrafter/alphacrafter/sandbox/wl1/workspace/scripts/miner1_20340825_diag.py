"""miner_1 2034-08-25: coverage diagnostics + library correlation for candidates."""
import numpy as np
import pandas as pd

panel = pd.read_pickle('scripts/panel_cache_20340825.pkl')
close = panel['close']; opn = panel['open']; high = panel['high']; low = panel['low']
ret = panel['ret']

print("per-asset close NaN counts (since 2021-01-01):")
sub = close[close.index >= '2021-01-01']
print(sub.isna().sum().to_dict())
print("\nper-asset open NaN counts:")
print(opn[opn.index >= '2021-01-01'].isna().sum().to_dict())
print("\nclose start date per asset:")
print(close.notna().idxmax().dt.date.to_dict())
print("\nopen start date per asset:")
print(opn.notna().idxmax().dt.date.to_dict())

# candidate signals
sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
bollz = (close - sma20) / std20
ddepth = 1.0 - close / close.rolling(60).max()
r2 = ret.clip(upper=0.0)**2
rsum = (ret**2).rolling(60).sum()
dskew = r2.rolling(60).sum() / rsum

# ---- library correlation (cross-sectional flattened, overlapping dates, LIVE cols) ----
LIVE = [c for c in close.columns if c not in ('HSI', 'CN10Y')]
lib_signals = {}
# reconstruct library factors from definitions
lib_signals['nclv_1d'] = -(close - low) / (high - low)
lib_signals['rev_2d'] = -(np.log(close) - np.log(close.shift(2)))
lib_signals['rev_5d'] = -(np.log(close) - np.log(close.shift(5)))
lib_signals['vol_of_vol20x60'] = close.pct_change().rolling(20).std().rolling(60).std()
mom = close.shift(5) / close.shift(125) - 1.0
lib_signals['mom_120d_skip5'] = mom
vix = panel['macro']['VIX']
vixr = vix.pct_change()
vixm = vix / vix.shift(20) - 1.0
def rolling_beta(y, x, w):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for col in y.columns:
        yy = y[col]; xx = x
        cov = yy.rolling(w).cov(xx); var = xx.rolling(w).var()
        out[col] = cov / var
    return out
asset_ret = close.pct_change()
beta60 = rolling_beta(asset_ret, vixr, 60)
lib_signals['vix_beta_cond_60x20'] = -beta60 * vixm

cands = {'bollz_20d': -bollz, 'ddepth_60d': ddepth, 'dskew_60d': -dskew, 'gap_rev_1d': -(opn/close.shift(1)-1)}
cands['xs_rev_5d'] = -(close/close.shift(5)-1.0).sub((close/close.shift(5)-1.0)[LIVE].mean(axis=1), axis=0)

idx = close.index[close.index >= '2021-01-01']
print("\nmax_abs_library_correlation per candidate (LIVE cols, flattened, overlapping):")
for cname, sig in cands.items():
    maxrho = 0.0; arg = None
    for lname, lsig in lib_signals.items():
        a = sig.loc[idx, LIVE].values.flatten()
        b = lsig.loc[idx, LIVE].values.flatten()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 500:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        if abs(rho) > abs(maxrho):
            maxrho = rho; arg = lname
    print(f"  {cname:12s} max|rho|={maxrho:+.3f} vs {arg}")

print("\ncoverage (LIVE, since 2021) per candidate:")
for cname, sig in cands.items():
    print(f"  {cname:12s} {float(sig.loc[idx, LIVE].notna().mean().mean()):.3f}")
