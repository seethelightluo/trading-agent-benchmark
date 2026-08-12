"""Screener: compute live factor values from price data and estimate recent cross-sectional IC."""
import pandas as pd
import numpy as np

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
CUT = pd.Timestamp('2029-10-08')

# ---- load closes ----
closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    dcol = 'date' if 'date' in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.sort_values(dcol)
    df = df[df[dcol] <= CUT]
    s = df.set_index(dcol)['close']
    closes[a] = s

px = pd.DataFrame(closes).sort_index()
rets = px.pct_change()
print('price panel:', px.shape, px.index[0].date(), '->', px.index[-1].date())

# ---- factor 1: dn_mkt_beta_60d (beta on downside equal-weight market) ----
mkt = rets.mean(axis=1)
down = mkt.where(mkt < 0, 0.0)
def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.Series(np.nan, index=y.index)
    xv = x.values; yv = y.values
    for i in range(win, len(y)):
        xw = xv[i-win:i]; yw = yv[i-win:i]
        mask = ~(np.isnan(xw) | np.isnan(yw))
        if mask.sum() < min_obs:
            continue
        xm = xw[mask]; ym = yw[mask]
        varx = np.var(xm)
        if varx == 0 or np.isnan(varx):
            continue
        out.iloc[i] = np.cov(xm, ym)[0, 1] / varx
    return out

dn_beta = pd.DataFrame({a: rolling_beta(rets[a], down) for a in ASSETS})

# ---- factor 2: rate_beta_cn10y_60d (beta on CN10Y pct change) ----
cn10y_ret = rets['CN10Y']
rate_beta = pd.DataFrame({a: rolling_beta(rets[a], cn10y_ret) for a in ASSETS})

# ---- factor 3: vol_adj_mom_accel_20x60 ----
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
vol20 = rets.rolling(20).std()
vol_adj = (mom20 - mom60) / vol20

factors = {
    'dn_mkt_beta_60d': dn_beta,
    'rate_beta_cn10y_60d': rate_beta,
    'vol_adj_mom_accel_20x60': vol_adj,
}
# persisted directions
direction = {'dn_mkt_beta_60d': 1, 'rate_beta_cn10y_60d': -1, 'vol_adj_mom_accel_20x60': 1}

# ---- forward 10d returns ----
fwd = px.shift(-10) / px - 1

# ---- rank IC ----
def rank_ic(fvals, fwd_ret, dates):
    ics = []
    for t in dates:
        f = fvals.loc[t]
        r = fwd_ret.loc[t]
        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            continue
        ic = f[valid].rank().corr(r[valid].rank())
        if pd.notna(ic):
            ics.append(ic)
    return pd.Series(ics)

all_dates = px.index
for label, win in [('last_125d', 125), ('last_250d', 250), ('full_2028_2029', 500)]:
    dates = all_dates[-win:]
    print(f'\n===== {label} ({dates[0].date()} .. {dates[-1].date()}) =====')
    for name, fvals in factors.items():
        ic = rank_ic(fvals, fwd, dates)
        if len(ic) == 0:
            print(f'  {name:24s} no valid IC dates'); continue
        signed_ic = ic.mean() * direction[name]
        print(f'  {name:24s} n={len(ic):4d} rawIC={ic.mean():+.4f} signedIC={signed_ic:+.4f} '
              f'ICIR={ic.mean()/ic.std():+.3f} hit={(ic>0).mean():.2f}')

# ---- cross-factor correlation (last 125d) ----
dates = all_dates[-125:]
print('\n===== factor cross-correlation (avg of daily cross-sectional corr, last 125d) =====')
names = list(factors)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        cors = []
        for t in dates:
            a = factors[names[i]].loc[t]; b = factors[names[j]].loc[t]
            v = a.notna() & b.notna()
            if v.sum() >= 5:
                c = a[v].corr(b[v])
                if pd.notna(c): cors.append(c)
        print(f'  {names[i]:22s} vs {names[j]:22s}: {np.mean(cors):+.3f} (n={len(cors)})')

# ---- forward-return dispersion (cross-sectional std of 10d fwd rets) ----
disp = fwd.std(axis=1).dropna()
print('\n===== cross-sectional dispersion of 10d fwd returns =====')
for label, win in [('last_125d', 125), ('last_250d', 250)]:
    d = disp.tail(win)
    print(f'  {label}: mean={d.mean():.3f} median={d.median():.3f} last={disp.iloc[-1]:.3f}')
