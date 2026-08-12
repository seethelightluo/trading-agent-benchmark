"""miner_1 cycle 2028-08-14: factor exploration batch Q.
Data visible window: <= 2028-08-11 (last completed trading day before decision date).
Screens NEW candidate factors + re-validates 3 library factors on the 15-asset cross-section.
No backtest/step usage; pure factor analytics.
"""
import pandas as pd, numpy as np, glob, os, json

CUT = '2028-08-11'
H = 10  # admission horizon (10 trading days, matching rebalance cadence)
MIN_VALID = 8

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
print('visible data range:', px.index.min().date(), '->', px.index.max().date(),
      '| rows', len(px), '| cols', len(px.columns))

# high/low/volume
HI, LO, VOL = {}, {}, {}
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    HI[sym] = df['high'].astype(float); LO[sym] = df['low'].astype(float)
    VOL[sym] = df['volume'].astype(float)
HI = pd.DataFrame(HI).sort_index(); HI = HI[HI.index <= CUT]
LO = pd.DataFrame(LO).sort_index(); LO = LO[LO.index <= CUT]
VOL = pd.DataFrame(VOL).sort_index(); VOL = VOL[VOL.index <= CUT]
print('volume null frac:', round(float(VOL.isna().mean().mean()), 4))

# macro observation-only signals (align to px index)
idx = {}
for sym in ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    idx[sym] = df['close'].astype(float).reindex(px.index)
idx = pd.DataFrame(idx).sort_index()

rets = px.pct_change()
mkt = rets.mean(axis=1)  # equal-weight 15-asset market proxy


def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.Series(index=y.index, dtype=float)
    ys_all = y.values.astype(float); xs_all = x.values.astype(float)
    for i in range(win, len(y)):
        ys, xs = ys_all[i-win:i], xs_all[i-win:i]
        m = ~(np.isnan(ys) | np.isnan(xs))
        if m.sum() < min_obs:
            continue
        if np.nanstd(xs[m]) < 1e-12:
            continue
        out.iloc[i] = np.polyfit(xs[m], ys[m], 1)[0]
    return out


def beta_frame(x_series):
    return pd.DataFrame({c: rolling_beta(rets[c], x_series, 60, 40) for c in px.columns}, index=px.index)


# ---------------- factor library (3 effective) ----------------
dnmkt = mkt.where(mkt < 0, 0.0)
F = {}
F['dn_mkt_beta_60d'] = beta_frame(dnmkt)
F['rate_beta_cn10y_60d'] = beta_frame(px['CN10Y'].pct_change())
r20 = px / px.shift(20) - 1
r60 = px / px.shift(60) - 1
v20 = rets.rolling(20).std()
F['vol_adj_mom_accel_20x60'] = (r20 - r60) / v20

# ---------------- NEW candidates batch Q ----------------
# 1) Kaufman efficiency ratio 20d: net move / total path
F['eff_ratio_20d'] = (px - px.shift(20)).abs() / rets.abs().rolling(20).sum()
# 2) drawdown depth from 60d high
F['dd_60d'] = px / px.rolling(60).max() - 1
# 3) crash asymmetry: 20d downside semideviation / 60d total vol
down_ret = rets.where(rets < 0, 0.0)
dstd20 = np.sqrt((down_ret ** 2).rolling(20).mean())
F['down_vol_ratio_20x60'] = dstd20 / rets.rolling(60).std()
# 4) skewness 20d
F['skew_20d'] = rets.rolling(20).skew()
# 5) excess kurtosis 20d
F['kurt_20d'] = rets.rolling(20).kurt()
# 6) gold beta 60d (safe-haven co-movement)
F['xau_beta_60d'] = beta_frame(px['XAU'].pct_change())
# 7) DXY beta 60d (USD strength sensitivity)
F['dxy_beta_60d'] = beta_frame(idx['DXY'].pct_change())
# 8) VIX beta 60d (risk-off sensitivity)
F['vix_beta_60d'] = beta_frame(idx['VIX'].pct_change())
# 9) Amihud illiquidity 20d: mean(|ret| / volume)
F['amihud_20d'] = (rets.abs() / VOL).rolling(20).mean()
# 10) win rate 60d: fraction of positive days
F['winrate_60d'] = (rets > 0).rolling(60).mean()
# 11) 5d return autocorrelation
F['autocorr_5d'] = rets.rolling(6).apply(lambda x: pd.Series(x).autocorr() if len(x) > 2 and np.nanstd(x) > 1e-12 else np.nan, raw=False)
# 12) intraday range ratio 20d: mean((high-low)/close)
F['range_ratio_20d'] = ((HI - LO) / px).rolling(20).mean()
# 13) BTC beta 60d (crypto sensitivity)
F['btc_beta_60d'] = beta_frame(px['BTC'].pct_change())
# 14) upside/downside capture 20d: mean(pos ret)/|mean(neg ret)|
pos = rets.where(rets > 0, np.nan)
neg = rets.where(rets < 0, np.nan)
F['updown_ratio_20d'] = pos.rolling(20).mean() / neg.rolling(20).mean().abs()
# 15) rolling sharpe 60d
F['sharpe_60d'] = rets.rolling(60).mean() / rets.rolling(60).std()
# 16) max drawdown 60d
roll_max = px.rolling(60).max()
F['maxdd_60d'] = (px / roll_max - 1).rolling(60).min()

# ---------------- IC engine ----------------
def ic_stats(fac, fwd, min_valid=MIN_VALID):
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
    return dict(ic=float(np.nanmean(ics)),
                icir=float(np.nanmean(ics) / np.nanstd(ics)) if np.nanstd(ics) > 0 else 0.0,
                hit=float(np.mean(ics > 0)), n=len(ics), ic_std=float(np.nanstd(ics)),
                first_date=str(dates[0].date()), last_date=str(dates[-1].date()))


print('\n=== FACTOR SCREEN (full history to %s, h=%d, min_valid=%d) ===' % (CUT, H, MIN_VALID))
results = {}
for name, fac in F.items():
    fwd = px.shift(-H) / px - 1
    s = ic_stats(fac, fwd)
    if s is None:
        print(f'{name:24s} NO VALID DATES'); continue
    cov_ad = float(fac.notna().mean().mean())
    cov_dates = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
    ranks = fac.rank(axis=1)
    to = float(ranks.diff().abs().mean().mean())
    results[name] = (s, cov_ad, cov_dates, to)
    flag = ' *** GATE PASS ***' if (abs(s['ic']) >= 0.0070 and abs(s['icir']) >= 0.0840) else ''
    print(f'{name:24s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:4d} '
          f'cov_ad={cov_ad:.3f} cov_d8={cov_dates:.3f} to_rank={to:.3f} {s["last_date"]}{flag}')

# decay for all candidates (h=1..20)
print('\n=== DECAY (IC by horizon) ===')
for name in F:
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwd = px.shift(-h) / px - 1
        s = ic_stats(F[name], fwd)
        decay[h] = round(s['ic'], 4) if s else np.nan
    print(f'{name:24s} {decay}')

# recent 250d window
print('\n=== RECENT 250d WINDOW (h=10) ===')
for name, fac in F.items():
    fac_r = fac.iloc[-250:]
    fwd = px.shift(-H) / px - 1
    fwd_r = fwd.iloc[-250:]
    s = ic_stats(fac_r, fwd_r)
    if s is None:
        continue
    flag = ' ***' if (abs(s['ic']) >= 0.0070 and abs(s['icir']) >= 0.0840) else ''
    print(f'{name:24s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:3d}{flag}')

# ---------------- library correlation (spearman, pairwise on valid rows) ----------------
lib = ['vol_adj_mom_accel_20x60', 'dn_mkt_beta_60d', 'rate_beta_cn10y_60d']
print('\n=== max |spearman corr| vs library (full period) ===')
for name, fac in F.items():
    mx = 0.0; arg = ''
    for lf in lib:
        if name == lf:
            continue
        dfc = pd.concat([fac.stack(), F[lf].stack()], axis=1).dropna()
        if len(dfc) < 100:
            continue
        rho = dfc[0].rank().corr(dfc[1].rank())
        if abs(rho) > mx:
            mx = abs(rho); arg = lf
    print(f'{name:24s} max_abs_corr={mx:.3f} vs {arg}')
