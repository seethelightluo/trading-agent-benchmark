"""miner_1 cycle 2029-01-15: factor exploration batch S + library revalidation.
Data visible through 2029-01-12 (last completed trading day before decision date).
Screens NEW candidate factors + re-validates 3 library factors on the 15-asset cross-section.
No backtest/step usage; pure factor analytics.
"""
import pandas as pd, numpy as np, glob, os, json

CUT = '2029-01-12'
H = 10  # admission horizon (10 trading days, matches rebalance cadence)
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

HI, LO, OP, VOL = {}, {}, {}, {}
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    HI[sym] = df['high'].astype(float); LO[sym] = df['low'].astype(float)
    OP[sym] = df['open'].astype(float); VOL[sym] = df['volume'].astype(float)
HI = pd.DataFrame(HI).sort_index(); HI = HI[HI.index <= CUT]
LO = pd.DataFrame(LO).sort_index(); LO = LO[LO.index <= CUT]
OP = pd.DataFrame(OP).sort_index(); OP = OP[OP.index <= CUT]
VOL = pd.DataFrame(VOL).sort_index(); VOL = VOL[VOL.index <= CUT]

# macro observation-only signals
idx = {}
for sym in ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    idx[sym] = df['close'].astype(float).reindex(px.index)
idx = pd.DataFrame(idx).sort_index()

rets = px.pct_change()
mkt = rets.mean(axis=1)


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

# ---------------- NEW candidates batch S ----------------
upmkt = mkt.where(mkt > 0, 0.0)
# S1 upside market beta 60d (complement to downside beta)
F['upside_beta_60d'] = beta_frame(upmkt)
# S2 yield-curve spread beta 60d: beta to change in (US10Y - CN10Y)
spread = px['US10Y'] - px['CN10Y']
F['yield_spread_beta_60d'] = beta_frame(spread.pct_change())
# S3 US10Y rate beta 60d (complement to CN10Y rate beta)
F['us10y_beta_60d'] = beta_frame(px['US10Y'].pct_change())
# S4 safe-haven beta 60d: beta to XAU
F['safe_haven_beta_60d'] = beta_frame(px['XAU'].pct_change())
# S5 efficiency ratio 60d: net move / total path (trend linearity)
F['efficiency_ratio_60d'] = (px - px.shift(60)).abs() / rets.abs().rolling(60).sum()
# S6 volume-price confirmation 20d: rolling corr(volume pct change, close pct change)
F['volume_price_corr_20d'] = rets.rolling(20).corr(VOL.pct_change())
# S7 60d range position: (close - min60)/(max60 - min60)
F['range_pos_60d'] = (px - px.rolling(60).min()) / (px.rolling(60).max() - px.rolling(60).min())
# S8 z-score 60d: (close - sma60)/std60
F['zscore_60d'] = (px - px.rolling(60).mean()) / px.rolling(60).std()
# S9 quarterly momentum skip 5 (63d mom, skip last 5)
F['mom63_skip5'] = px.shift(5) / px.shift(68) - 1
# S10 downside/upside vol ratio 60d (return asymmetry)
down_ret = rets.where(rets < 0, 0.0)
up_ret = rets.where(rets > 0, 0.0)
F['down_up_vol_ratio_60d'] = np.sqrt((down_ret ** 2).rolling(60).mean()) / np.sqrt((up_ret ** 2).rolling(60).mean())
# S11 WTI (energy) beta 60d
F['wti_beta_60d'] = beta_frame(px['WTI'].pct_change())
# S12 DXY (USD) beta 60d
F['dxy_beta_60d'] = beta_frame(idx['DXY'].pct_change())
# S13 VIX sensitivity 60d: rolling corr(asset ret, VIX change)
F['vix_corr_60d'] = rets.rolling(60).corr(idx['VIX'].pct_change())
# S14 vol-adjusted 5d reversal: -mom5 / vol5
F['mean_rev_5d_voladj'] = -(px / px.shift(5) - 1) / rets.rolling(5).std()
# S15 30d drawdown depth: close / rolling_max(close,30) - 1
F['drawdown_30d'] = px / px.rolling(30).max() - 1
# S16 correlation to equal-weight market 60d (diversification/co-movement)
F['mkt_corr_60d'] = rets.rolling(60).corr(mkt)


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
        print(f'{name:26s} NO VALID DATES'); continue
    cov_ad = float(fac.notna().mean().mean())
    cov_dates = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
    ranks = fac.rank(axis=1)
    to = float(ranks.diff().abs().mean().mean())
    results[name] = (s, cov_ad, cov_dates, to)
    flag = ' *** GATE PASS ***' if (abs(s['ic']) >= 0.0070 and abs(s['icir']) >= 0.0840) else ''
    print(f'{name:26s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:4d} '
          f'cov_ad={cov_ad:.3f} cov_d8={cov_dates:.3f} to_rank={to:.3f} {s["last_date"]}{flag}')

print('\n=== DECAY (IC by horizon) ===')
for name in F:
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwd = px.shift(-h) / px - 1
        s = ic_stats(F[name], fwd)
        decay[h] = round(s['ic'], 4) if s else np.nan
    print(f'{name:26s} {decay}')

print('\n=== RECENT 250d WINDOW (h=10) ===')
for name, fac in F.items():
    fac_r = fac.iloc[-250:]
    fwd = px.shift(-H) / px - 1
    fwd_r = fwd.iloc[-250:]
    s = ic_stats(fac_r, fwd_r)
    if s is None:
        continue
    flag = ' ***' if (abs(s['ic']) >= 0.0070 and abs(s['icir']) >= 0.0840) else ''
    print(f'{name:26s} IC={s["ic"]:+.4f} ICIR={s["icir"]:+.3f} hit={s["hit"]:.3f} n={s["n"]:3d}{flag}')

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
    print(f'{name:26s} max_abs_corr={mx:.3f} vs {arg}')

# ---------------- yearly IC breakdown for gate-passing candidates ----------------
print('\n=== YEARLY IC (h=10) for candidates passing full gate ===')
for name, fac in F.items():
    fwd = px.shift(-H) / px - 1
    s = ic_stats(fac, fwd)
    if s is None or not (abs(s['ic']) >= 0.0070 and abs(s['icir']) >= 0.0840):
        continue
    fac_y = fac.groupby(fac.index.year)
    out = {}
    for yr, grp in fac_y:
        fwd_y = fwd.loc[grp.index]
        ss = ic_stats(grp, fwd_y)
        out[yr] = (round(ss['ic'], 4), round(ss['icir'], 3), ss['n']) if ss else None
    print(f'{name:26s} {out}')
