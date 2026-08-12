"""miner_1 cycle 2028-06-19: factor validation harness.
Data visible window: <= 2028-06-16 (last completed trading day before decision date 2028-06-19).
Re-validates the 3 effective library factors + screens new candidates on the 15-asset cross-section.
No backtest/step usage; pure factor analytics.
"""
import pandas as pd, numpy as np, glob, os, json, hashlib, base64, zlib

CUT = '2028-06-16'
H = 10  # admission horizon (10 trading days, matching rebalance cadence)

# ---------------- load 15 tradable assets ----------------
files = sorted(glob.glob('../persistent/stock_data/*.csv'))
px = {}
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[sym] = df['close'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[px.index <= CUT]
print('visible data range:', px.index.min().date(), '->', px.index.max().date(), '| rows', len(px), '| cols', len(px.columns))

rets = px.pct_change()
mkt = rets.mean(axis=1)  # equal-weight 15-asset market proxy

# ---------------- factor library ----------------
def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.Series(index=y.index, dtype=float)
    ys_all = y.values.astype(float); xs_all = x.values.astype(float)
    idx = y.index
    for i in range(win, len(y)):
        ys, xs = ys_all[i-win:i], xs_all[i-win:i]
        m = ~(np.isnan(ys) | np.isnan(xs))
        if m.sum() < min_obs:
            continue
        if np.nanstd(xs[m]) < 1e-12:
            continue
        out.iloc[i] = np.polyfit(xs[m], ys[m], 1)[0]
    return out

def make_factors(px, rets, mkt):
    F = {}
    # --- existing library factors ---
    dnmkt = mkt.where(mkt < 0, 0.0)
    F['dn_mkt_beta_60d'] = pd.DataFrame({c: rolling_beta(rets[c], dnmkt, 60, 40) for c in px.columns}, index=px.index)
    F['rate_beta_cn10y_60d'] = pd.DataFrame({c: rolling_beta(rets[c], px['CN10Y'].pct_change(), 60, 40) for c in px.columns}, index=px.index)
    r20 = px / px.shift(20) - 1
    r60 = px / px.shift(60) - 1
    v20 = rets.rolling(20).std()
    F['vol_adj_mom_accel_20x60'] = (r20 - r60) / v20
    # --- new candidates ---
    # 1) drawdown depth 20d (distance below recent high): (close/rolling_max(close,20) - 1)
    F['dd_20d'] = px / px.rolling(20).max() - 1
    # 2) rolling sharpe 60d
    F['sharpe_60d'] = rets.rolling(60).mean() / rets.rolling(60).std()
    # 3) downside-vol ratio: 20d downside std / 60d total std (crash asymmetry)
    down_ret = rets.where(rets < 0, 0.0)
    dstd20 = np.sqrt((down_ret.rolling(20).apply(lambda x: np.nanmean(x**2), raw=True)))
    tstd60 = rets.rolling(60).std()
    F['down_vol_ratio_20x60'] = dstd20 / tstd60
    # 4) range position 20d: (close - min(low,20)) / (max(high,20) - min(low,20)) -- needs high/low
    return F

# need high/low for range factor; load separately
def load_ohlc():
    files = sorted(glob.glob('../persistent/stock_data/*.csv'))
    hi, lo = {}, {}
    for f in files:
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f); df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
        hi[sym] = df['high'].astype(float); lo[sym] = df['low'].astype(float)
    return pd.DataFrame(hi).sort_index(), pd.DataFrame(lo).sort_index()

H_, L_ = load_ohlc()
H_ = H_[H_.index <= CUT]; L_ = L_[L_.index <= CUT]

F = make_factors(px, rets, mkt)
F['range_pos_20d'] = (px - L_.rolling(20).min()) / (H_.rolling(20).max() - L_.rolling(20).min())
F['vol_term_10x60'] = rets.rolling(10).std() / rets.rolling(60).std()
F['skew_20d'] = rets.rolling(20).skew()
# wti beta (commodity sensitivity)
F['wti_beta_60d'] = pd.DataFrame({c: rolling_beta(rets[c], px['WTI'].pct_change(), 60, 40) for c in px.columns}, index=px.index)

# ---------------- IC engine ----------------
def ic_stats(fac, fwd, min_valid=8):
    dates, ics = [], []
    fac_a, fwd_a = fac.values, fwd.values
    for i in range(len(fac)):
        fv, rv = fac_a[i], fwd_a[i]
        m = ~(np.isnan(fv) | np.isnan(rv))
        if m.sum() < min_valid:
            continue
        if np.nanstd(fv[m]) < 1e-12 or np.nanstd(rv[m]) < 1e-12:
            continue
        ics.append(pd.Series(fv[m]).rank().corr(pd.Series(rv[m]).rank()))
        dates.append(fac.index[i])
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    return dict(ic=float(np.nanmean(ics)), icir=float(np.nanmean(ics)/np.nanstd(ics)) if np.nanstd(ics) > 0 else 0.0,
                hit=float(np.mean(ics > 0)), n=len(ics), ic_std=float(np.nanstd(ics)))

print('\n=== FACTOR SCREEN (full history to %s, h=%d, min_valid=8) ===' % (CUT, H))
results = {}
for name, fac in F.items():
    fwd = px.shift(-H) / px - 1
    s = ic_stats(fac, fwd)
    if s is None:
        print(f'{name:24s} NO VALID DATES'); continue
    # coverage
    cov_ad = float(fac.notna().mean().mean())
    cov_dates = float((fac.notna().sum(axis=1) >= 8).mean())
    # turnover: mean abs change of cross-sectional rank between consecutive valid dates
    ranks = fac.rank(axis=1)
    to = float(ranks.diff().abs().mean().mean())
    results[name] = (s, cov_ad, cov_dates, to)
    print(f'{name:24s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:4d} '
          f'cov_ad={cov_ad:.3f} cov_d8={cov_dates:.3f} to_rank={to:.3f}')

# decay for top candidates + library
print('\n=== DECAY (IC by horizon) ===')
for name in ['dn_mkt_beta_60d', 'rate_beta_cn10y_60d', 'vol_adj_mom_accel_20x60', 'dd_20d', 'sharpe_60d', 'range_pos_20d', 'vol_term_10x60', 'skew_20d']:
    if name not in F:
        continue
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwd = px.shift(-h) / px - 1
        s = ic_stats(F[name], fwd)
        decay[h] = round(s['ic'], 4) if s else np.nan
    print(f'{name:24s} decay={decay}')

# recent 250d window
print('\n=== RECENT 250d WINDOW (h=10) ===')
for name, fac in F.items():
    fac_r = fac.iloc[-250:]
    fwd = px.shift(-H) / px - 1
    fwd_r = fwd.iloc[-250:]
    s = ic_stats(fac_r, fwd_r)
    if s is None:
        continue
    print(f'{name:24s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:3d}')
