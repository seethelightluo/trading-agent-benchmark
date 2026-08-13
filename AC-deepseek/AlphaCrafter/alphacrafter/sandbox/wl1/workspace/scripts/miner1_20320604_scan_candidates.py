"""miner1 2032-06-04: scan new factor candidates across cross-asset universe."""
import pandas as pd, numpy as np, time

t0 = time.time()
panel = pd.read_pickle('scripts/panel_cache_20320604.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']
open_ = panel['open']; high = panel['high']; low = panel['low']
macro = panel['macro']
print(f"panel loaded {time.time()-t0:.1f}s", flush=True)

def ic_series_vec(signal, close_, hz=1):
    sig_r = signal.rank(axis=1, pct=True).values
    fwd = (np.log(close_.shift(-hz) / close_)).values
    dates = signal.index
    ics = []
    for t in range(len(signal) - hz):
        s = sig_r[t]; f = fwd[t]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() >= 8:
            ic = np.corrcoef(s[m], f[m])[0, 1] if m.sum() < len(s) else np.corrcoef(s, f)[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    return pd.Series(ics, index=dates[:len(ics)])

def stats(ics):
    if len(ics) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    sd = float(ics.std(ddof=1))
    return dict(ic=float(ics.mean()), icir=float(ics.mean()/sd) if sd > 0 else np.nan,
                hit=float((ics > 0).mean()), n=len(ics))

# ---------- candidate signals ----------
sig = {}

# 1. Relative momentum: own return minus cross-sectional mean return (idiosyncratic trend)
for k in [20, 60, 120]:
    mom = np.log(close) - np.log(close.shift(k))
    sig[f'rel_mom_{k}d'] = mom - mom.mean(axis=1)

# 2. Kaufman efficiency ratio: |net move| / sum of absolute moves
for k in [20, 60]:
    net = (close - close.shift(k)).abs()
    path = ret.abs().rolling(k).sum()
    sig[f'eff_ratio_{k}d'] = net / path.replace(0, np.nan)

# 3. Upside/downside capture: mean up-day ret / mean |down-day ret|
for k in [20, 60]:
    up = ret.where(ret > 0, 0.0).rolling(k).mean()
    dn = ret.where(ret < 0, 0.0).rolling(k).mean().abs()
    sig[f'updn_{k}d'] = up / dn.replace(0, np.nan)

# 4. Rolling skewness of daily returns
for k in [20, 60]:
    sig[f'skew_{k}d'] = ret.rolling(k).skew()

# 5. Range position over window: (close - min) / (max - min)
for k in [5, 10]:
    hi = close.rolling(k).max(); lo = close.rolling(k).min()
    sig[f'range_pos_{k}d'] = (close - lo) / (hi - lo).replace(0, np.nan)

# 6. Volatility-scaled momentum (trend Sharpe)
for k in [20, 60]:
    mom = np.log(close) - np.log(close.shift(k))
    rv = ret.rolling(k).std()
    sig[f'mom_sharpe_{k}d'] = mom / (rv * np.sqrt(k)).replace(0, np.nan)

# 7. Positive-day fraction / trend consistency
for k in [10, 20]:
    sig[f'posfrac_{k}d'] = (ret > 0).rolling(k).mean()

# 8. DXY beta (risk-off dollar sensitivity), direction decided by IC sign
dxy_r = np.log(macro['DXY']).diff()
for win in [60]:
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        a = ret[c]; b = dxy_r
        out[c] = a.rolling(win).cov(b) / b.rolling(win).var().replace(0, np.nan)
    sig[f'dxy_beta_{win}d'] = out

# 9. US10Y-beta (duration sensitivity)
u10_r = np.log(close['US10Y']).diff()
for win in [60]:
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        a = ret[c]; b = u10_r
        out[c] = a.rolling(win).cov(b) / b.rolling(win).var().replace(0, np.nan)
    sig[f'u10y_beta_{win}d'] = out

print(f"signals built {time.time()-t0:.1f}s", flush=True)

# ---------- evaluate ----------
rows = []
for name, s in sig.items():
    s = s.reindex(close.index)
    full1 = stats(ic_series_vec(s, close, 1))
    full5 = stats(ic_series_vec(s, close, 5))
    r2 = stats(ic_series_vec(s.loc[close.index[-520:]], close.loc[close.index[-520:]], 1))
    r1 = stats(ic_series_vec(s.loc[close.index[-260:]], close.loc[close.index[-260:]], 1))
    cov = float(s.notna().mean().mean())
    rp = s.rank(axis=1, pct=True)
    turn = float(rp.diff().abs().mean().mean())
    rows.append((name, full1['ic'], full1['icir'], full1['hit'], full1['n'],
                 full5['ic'], full5['icir'], r2['ic'], r2['icir'], r1['ic'], r1['icir'],
                 cov, turn))
    print(f"{name:18s} FULL ic1={full1['ic']:+.4f} icir1={full1['icir']:+.3f} hit1={full1['hit']:.3f} "
          f"ic5={full5['ic']:+.4f} icir5={full5['icir']:+.3f} | 2Y ic1={r2['ic']:+.4f} icir1={r2['icir']:+.3f} "
          f"| 1Y ic1={r1['ic']:+.4f} icir1={r1['icir']:+.3f} | cov={cov:.3f} turn={turn:.3f}", flush=True)

print(f"\ntotal {time.time()-t0:.1f}s")
