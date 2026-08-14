"""SCREENER factor IC & correlation check as of 2035-04-27 (visible through)."""
import pandas as pd, numpy as np

ASSETS = ['000300.SH','000688.SH','SPX','NDX','SOX','HSI','N225','SX5E','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
LIVE = ['000688.SH','SPX','NDX','SOX','N225','SX5E','XAU','COPPER','WTI','US10Y']
END = '2035-04-27'

closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].reset_index(drop=True)
    closes[a] = pd.Series(df['close'].values, index=df['date'])
px = pd.DataFrame(closes).sort_index()
rets = px.pct_change()

# correct beta: covariance of asset ret with market factor / variance of factor
def rolling_beta(y, x, win=60):
    # y: DataFrame, x: Series -> DataFrame of betas
    xy = pd.concat([y, x.rename('_f')], axis=1)
    cov = xy.rolling(win).cov()
    out = {}
    for a in y.columns:
        c = cov.loc[(slice(None), a), '_f'] if hasattr(cov.index,'levels') else None
        # simpler: loop
        out[a] = y[a].rolling(win).cov(x) / x.rolling(win).var()
    return pd.DataFrame(out)

fwd = 10
fwd_ret = px.pct_change(fwd).shift(-fwd)  # forward h-day return

def rank_ic_series(factor):
    ics = []
    dates = []
    for i in range(len(factor)-fwd):
        f = factor.iloc[i]
        r = fwd_ret.iloc[i]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            ics.append(f[valid].rank().corr(r[valid].rank()))
            dates.append(factor.index[i])
    return pd.Series(ics, index=dates)

# factor 1
mom20 = px/px.shift(20)-1
mom60 = px/px.shift(60)-1
vol20 = rets.rolling(20).std()
f1 = (mom20 - mom60)/vol20

# factor 2: dn beta on down-market days (EW market, min(ret,0))
mkt = rets[LIVE].mean(axis=1)
mkt_neg = mkt.clip(upper=0)
f2 = pd.DataFrame({a: rets[a].rolling(60).cov(mkt_neg)/mkt_neg.rolling(60).var() for a in ASSETS})

# factor 3: rate beta vs CN10Y change
dcn = px['CN10Y'].pct_change()
f3 = pd.DataFrame({a: rets[a].rolling(60).cov(dcn)/dcn.rolling(60).var() for a in ASSETS})

print("=== factor rank-IC by trailing window (h=10) ===")
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    s = rank_ic_series(f)
    print(name, "n=", len(s))
    for look in [30, 60, 120, 250]:
        sub = s.iloc[-look:]
        if len(sub) >= 5:
            ic = sub.mean(); sd = sub.std(); icir = ic/sd if sd > 0 else np.nan
            print(f"  {look}d: IC={ic:+.4f} ICIR={icir:+.3f} pos%={100*(sub>0).mean():.0f}")
        else:
            print(f"  {look}d: n<5")

print("\n=== factor correlation (last 120d daily factor values) ===")
fac = pd.concat([f1.rename(columns=lambda c: c+'_f1'),
                 f2.rename(columns=lambda c: c+'_f2'),
                 f3.rename(columns=lambda c: c+'_f3')], axis=1)
corr = fac.iloc[-120:].corr()
# average cross-factor corr on live names
f1l, f2l, f3l = f1[LIVE].iloc[-120:], f2[LIVE].iloc[-120:], f3[LIVE].iloc[-120:]
c12 = np.nanmean([f1l[a].corr(f2l[a]) for a in LIVE])
c13 = np.nanmean([f1l[a].corr(f3l[a]) for a in LIVE])
c23 = np.nanmean([f2l[a].corr(f3l[a]) for a in LIVE])
print(f"avg pairwise corr (live names): f1-f2={c12:+.2f} f1-f3={c13:+.2f} f2-f3={c23:+.2f}")

print("\n=== pairwise return correlation 60d (live names) ===")
r60 = rets[LIVE].iloc[-60:]
c = r60.corr()
mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
print("avg:", round(c.values[mask].mean(),3), "median:", round(np.median(c.values[mask]),3))

print("\n=== live-name 20d momentum cross-section (for context) ===")
m20 = (px[LIVE].iloc[-1]/px[LIVE].iloc[-21]-1)*100
print(m20.sort_values(ascending=False).round(2).to_string())

print("\n=== last 3 months IC by month (vol_adj_mom_accel) ===")
s1 = rank_ic_series(f1)
s1.index = pd.to_datetime(s1.index)
print(s1.resample('ME').agg(['mean','count']).round(3).tail(6).to_string())
