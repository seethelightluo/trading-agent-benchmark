"""SCREENER analysis for cycle 2035-08-06 (visible through 2035-08-03)."""
import pandas as pd, numpy as np

ASSETS = ['000300.SH','000688.SH','SPX','NDX','SOX','HSI','N225','SX5E','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
LIVE = ['000688.SH','SPX','NDX','SOX','N225','SX5E','XAU','COPPER','WTI','US10Y']
END = '2035-08-03'

closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].reset_index(drop=True)
    closes[a] = pd.Series(df['close'].values, index=df['date'])
px = pd.DataFrame(closes).sort_index()
rets = px.pct_change()

print("=== data spans ===")
print(px.index.min(), '->', px.index.max(), 'rows', len(px))

print("\n=== per-asset regime stats (through", END, ") ===")
rows = []
for a in ASSETS:
    s = px[a]; r = rets[a]
    mom20 = s.iloc[-1]/s.iloc[-21]-1 if len(s) > 21 else np.nan
    mom60 = s.iloc[-1]/s.iloc[-61]-1 if len(s) > 61 else np.nan
    mom120 = s.iloc[-1]/s.iloc[-121]-1 if len(s) > 121 else np.nan
    vol20 = r.iloc[-20:].std()*np.sqrt(252)
    vol60 = r.iloc[-60:].std()*np.sqrt(252)
    dd60 = (s.iloc[-60:]/s.iloc[-60:].cummax()-1).min() if len(s) > 60 else np.nan
    ma20 = s.iloc[-20:].mean(); ma60 = s.iloc[-60:].mean() if len(s) > 60 else np.nan
    up5 = (r.iloc[-5:] > 0).sum()
    rows.append(dict(asset=a, last=round(s.iloc[-1],2), mom20=round(mom20*100,2), mom60=round(mom60*100,2),
                     mom120=round(mom120*100,2), vol20=round(vol20*100,1), vol60=round(vol60*100,1),
                     dd60=round(dd60*100,1), above_ma20=last>ma20 if False else s.iloc[-1]>ma20,
                     above_ma60=s.iloc[-1]>ma60, up5=up5))
reg = pd.DataFrame(rows)
print(reg.to_string(index=False))

print("\n=== cross-sectional stats ===")
print("median 20d mom:", round(reg.mom20.median(),2), " dispersion(20d mom std):", round(reg.mom20.std(),2))
print("median 60d mom:", round(reg.mom60.median(),2), " dispersion(60d mom std):", round(reg.mom60.std(),2))
print("median 20d vol:", round(reg.vol20.median(),1), " mean 20d vol:", round(reg.vol20.mean(),1))
print("assets above MA20:", int(reg.above_ma20.sum()), "/15  above MA60:", int(reg.above_ma60.sum()), "/15")

r60 = rets.iloc[-60:]
c = r60.corr()
mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
print("\navg pairwise corr 60d (all 15):", round(c.values[mask].mean(),3), " median:", round(np.median(c.values[mask]),3))
r60l = rets[LIVE].iloc[-60:]
cl = r60l.corr()
maskl = np.triu(np.ones(cl.shape, dtype=bool), k=1)
print("avg pairwise corr 60d (live 10):", round(cl.values[maskl].mean(),3))

mkt = rets.mean(axis=1)
mkt_lev = (1+mkt).cumprod()
mkt_ma20 = mkt_lev.rolling(20).mean()
mkt_ma60 = mkt_lev.rolling(60).mean()
print("\nEW-market 20d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-21]-1)*100,2),
      " 60d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-61]-1)*100,2),
      " 120d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-121]-1)*100,2))
print("EW-market above MA20:", mkt_lev.iloc[-1] > mkt_ma20.iloc[-1], " above MA60:", mkt_lev.iloc[-1] > mkt_ma60.iloc[-1])
sign = np.sign(mkt.iloc[-15:])
streak = 0
for v in sign[::-1]:
    if v > 0: streak += 1
    else: break
print("recent up-day streak:", streak, "of last 15; last 5d mkt:", [round(x*100,2) for x in mkt.iloc[-5:]])
print("EW-market 20d vol ann:", round(mkt.iloc[-20:].std()*np.sqrt(252)*100,1),
      " 60d vol ann:", round(mkt.iloc[-60:].std()*np.sqrt(252)*100,1))

# ---- factor signals & recent IC ----
def rank_ic_series(factor, fwd=10):
    fwd_ret = px.pct_change(fwd).shift(-fwd)
    ics, dates = [], []
    for i in range(len(factor)-fwd):
        f = factor.iloc[i]; r = fwd_ret.iloc[i]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            ics.append(f[valid].rank().corr(r[valid].rank()))
            dates.append(factor.index[i])
    return pd.Series(ics, index=dates)

mom20 = px/px.shift(20)-1
mom60 = px/px.shift(60)-1
vol20 = rets.rolling(20).std()
f1 = (mom20 - mom60)/vol20
mkt_neg = mkt.clip(upper=0)
f2 = pd.DataFrame({a: rets[a].rolling(60).cov(mkt_neg)/mkt_neg.rolling(60).var() for a in ASSETS})
dcn = px['CN10Y'].pct_change()
f3 = pd.DataFrame({a: rets[a].rolling(60).cov(dcn)/dcn.rolling(60).var() for a in ASSETS})

print("\n=== factor rank-IC (h=10) trailing windows ===")
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    s = rank_ic_series(f)
    print(name, "n=", len(s))
    for look in [30, 60, 120, 250]:
        sub = s.iloc[-look:]
        if len(sub) >= 5:
            ic = sub.mean(); sd = sub.std(); icir = ic/sd if sd > 0 else np.nan
            print(f"  {look}d: IC={ic:+.4f} ICIR={icir:+.3f} pos%={100*(sub>0).mean():.0f} n={len(sub)}")
        else:
            print(f"  {look}d: n<5")

print("\n=== factor correlation (last 120d, live names avg) ===")
f1l, f2l, f3l = f1[LIVE].iloc[-120:], f2[LIVE].iloc[-120:], f3[LIVE].iloc[-120:]
c12 = np.nanmean([f1l[a].corr(f2l[a]) for a in LIVE])
c13 = np.nanmean([f1l[a].corr(f3l[a]) for a in LIVE])
c23 = np.nanmean([f2l[a].corr(f3l[a]) for a in LIVE])
print(f"avg pairwise corr (live names): f1-f2={c12:+.2f} f1-f3={c13:+.2f} f2-f3={c23:+.2f}")

print("\n=== latest cross-section factor values (2035-08-03) ===")
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    last = f.iloc[-1].dropna()
    print(name, "valid:", len(last), "| top:", last.nlargest(4).index.tolist(), "| bottom:", last.nsmallest(4).index.tolist())

print("\n=== live-name 20d momentum cross-section ===")
m20 = (px[LIVE].iloc[-1]/px[LIVE].iloc[-21]-1)*100
print(m20.sort_values(ascending=False).round(2).to_string())

print("\n=== monthly IC (vol_adj_mom_accel) last 8 months ===")
s1 = rank_ic_series(f1)
s1.index = pd.to_datetime(s1.index)
print(s1.resample('ME').agg(['mean','count']).round(3).tail(8).to_string())

print("\n=== macro obs (index_data) ===")
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    try:
        df = pd.read_csv(f'../persistent/index_data/{m}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= END]
        s = df['close'] if 'close' in df.columns else df['value']
        last = s.iloc[-1]
        m20 = last/s.iloc[-21]-1 if len(s)>21 else np.nan
        m60 = last/s.iloc[-61]-1 if len(s)>61 else np.nan
        print(f"{m}: last={last:.3f} 20d={m20*100:+.2f}% 60d={m60*100:+.2f}%  (rows {len(s)})")
    except Exception as e:
        print(m, "ERR", e)
