import numpy as np, pandas as pd

DATA = '../persistent/stock_data'
WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = '2035-02-28'

close = {}
for s in WATCH:
    df = pd.read_csv(f'{DATA}/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF]
    df = df.set_index('date').sort_index()
    close[s] = df['close']
close = pd.DataFrame(close).replace(0, np.nan).ffill()
ret = close.pct_change()
print('assets:', len(WATCH), 'dates:', len(close), 'range', close.index[0].date(), '->', close.index[-1].date())

H = 10
fwd = close.pct_change(H).shift(-H)

candidates = {}

# B: distance from rolling max 60d
candidates['dd_from_high_60'] = close / close.rolling(60).max()
# C: z-distance from rolling mean 60
candidates['z_dist_mean_60'] = (close - close.rolling(60).mean()) / close.rolling(60).std()
# D: RSI 14
delta = close.diff()
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
candidates['rsi_14'] = 100 - 100/(1 + up/dn.replace(0,np.nan))
# E: downside ratio 20
down = ret.clip(upper=0)
downsd = (down**2).rolling(20).mean()**0.5
totstd = ret.rolling(20).std()
candidates['downside_ratio_20'] = downsd / totstd.replace(0,np.nan)
# A: Amihud illiquidity 20 = mean(|ret|/close)
illiq = (ret.abs()/close).replace([np.inf,-np.inf],np.nan).rolling(20).mean()
candidates['illiq_retclose20'] = illiq
# G: cross-sectional relative momentum 20
cs = ret.rolling(20).mean()*20
candidates['rel_mom_20'] = cs - cs.mean(axis=1).to_frame()[0]
# H: realized vol 20 (test low-vol premium)
candidates['rv_20'] = ret.rolling(20).std()
# I: 3-day reversal
candidates['rev_3d'] = -ret.rolling(3).sum()
# J: 60d skewness
candidates['skew60'] = ret.rolling(60).skew()
# K: CS beta 20 (to cs mean return)
cs_mean = ret.mean(axis=1)
cov = ret.rolling(20).cov(cs_mean)
var = cs_mean.rolling(20).var()
candidates['cs_beta_20'] = cov / var.replace(0,np.nan)

for name, fac in candidates.items():
    ics = []
    for d in fac.index:
        s = fac.loc[d]; r = fwd.loc[d]
        m = s.notna() & r.notna()
        if m.sum() >= 8:
            ic = np.corrcoef(s[m], r[m])[0,1]
            if np.isfinite(ic): ics.append((d, ic))
    if not ics:
        print(f'{name}: no valid IC dates'); continue
    icv = np.array([x[1] for x in ics])
    ic = icv.mean(); icir = icv.mean()/icv.std(ddof=1) if icv.std(ddof=1)>0 else np.nan
    hit = np.mean(icv>0)
    avg_cov = np.nanmean([fac.loc[d].notna().mean() for d,_ in ics])
    print(f'{name}: n={len(icv)} IC={ic:+.4f} ICIR={icir:+.3f} hit={hit:.3f} coverage={avg_cov:.2f}')