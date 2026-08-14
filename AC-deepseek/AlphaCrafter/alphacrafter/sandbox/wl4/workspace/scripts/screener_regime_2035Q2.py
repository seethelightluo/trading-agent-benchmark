"""SCREENER regime assessment as of visible-through 2035-04-27 (current 2035-04-30)."""
import pandas as pd, numpy as np, glob, json

ASSETS = ['000300.SH','000688.SH','SPX','NDX','SOX','HSI','N225','SX5E','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = '2035-04-27'

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
    s = px[a]
    r = rets[a]
    mom20 = s.iloc[-1]/s.iloc[-21]-1 if len(s) > 21 else np.nan
    mom60 = s.iloc[-1]/s.iloc[-61]-1 if len(s) > 61 else np.nan
    mom120 = s.iloc[-1]/s.iloc[-121]-1 if len(s) > 121 else np.nan
    vol20 = r.iloc[-20:].std()*np.sqrt(252)
    vol60 = r.iloc[-60:].std()*np.sqrt(252)
    dd60 = (s.iloc[-60:]/s.iloc[-60:].cummax()-1).min() if len(s) > 60 else np.nan
    ma20 = s.iloc[-20:].mean(); ma60 = s.iloc[-60:].mean() if len(s) > 60 else np.nan
    up5 = (r.iloc[-5:] > 0).sum()
    last = s.iloc[-1]
    rows.append(dict(asset=a, last=round(last,2), mom20=round(mom20*100,2), mom60=round(mom60*100,2),
                     mom120=round(mom120*100,2), vol20=round(vol20*100,1), vol60=round(vol60*100,1),
                     dd60=round(dd60*100,1), above_ma20=last>ma20, above_ma60=last>ma60, up5=up5))
reg = pd.DataFrame(rows)
print(reg.to_string(index=False))

print("\n=== cross-sectional stats ===")
print("median 20d mom:", round(reg.mom20.median(),2), " dispersion(20d mom std):", round(reg.mom20.std(),2))
print("median 60d mom:", round(reg.mom60.median(),2), " dispersion(60d mom std):", round(reg.mom60.std(),2))
print("median 20d vol:", round(reg.vol20.median(),1), " mean 20d vol:", round(reg.vol20.mean(),1))
print("assets above MA20:", int(reg.above_ma20.sum()), "/15  above MA60:", int(reg.above_ma60.sum()), "/15")

# average pairwise correlation of daily returns (60d window, trailing)
r60 = rets.iloc[-60:]
c = r60.corr()
mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
print("\navg pairwise corr 60d:", round(c.values[mask].mean(),3), " median:", round(np.median(c.values[mask]),3))

# equal-weight market trend & strength
mkt = rets.mean(axis=1)
mkt_lev = (1+mkt).cumprod()
mkt_ma20 = mkt_lev.rolling(20).mean()
mkt_ma60 = mkt_lev.rolling(60).mean()
print("\nEW-market 20d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-21]-1)*100,2),
      " 60d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-61]-1)*100,2) if len(mkt_lev)>61 else None,
      " 120d ret:", round((mkt_lev.iloc[-1]/mkt_lev.iloc[-121]-1)*100,2) if len(mkt_lev)>121 else None)
print("EW-market above MA20:", mkt_lev.iloc[-1] > mkt_ma20.iloc[-1], " above MA60:", mkt_lev.iloc[-1] > mkt_ma60.iloc[-1])
# up-day streak
sign = np.sign(mkt.iloc[-15:])
streak = 0
for v in sign[::-1]:
    if v > 0: streak += 1
    else: break
print("recent up-day streak:", streak, "of last 15; last 5d mkt:", [round(x*100,2) for x in mkt.iloc[-5:]])

# 60d realized vol of EW market (annualized)
print("EW-market 20d vol ann:", round(mkt.iloc[-20:].std()*np.sqrt(252)*100,1),
      " 60d vol ann:", round(mkt.iloc[-60:].std()*np.sqrt(252)*100,1))

# ---- factor signal replication & recent IC ----
def rank_ic(factor, fwd=10):
    """cross-sectional rank IC of factor vs fwd return, trailing windows"""
    out = {}
    for look in [60, 120, 250]:
        sub = factor.iloc[-look:]
        ics = []
        for i in range(len(sub)-fwd):
            f = sub.iloc[i]
            r = px.pct_change(fwd).iloc[i+fwd]
            valid = f.notna() & r.notna()
            if valid.sum() >= 8:
                ics.append(f[valid].rank().corr(r[valid].rank()))
        out[look] = (np.nanmean(ics) if ics else np.nan, np.nanstd(ics) if ics else np.nan, len(ics))
    return out

fwd = 10
# factor 1: vol_adj_mom_accel_20x60 = (mom20 - mom60)/vol20
mom20 = px/px.shift(20)-1
mom60 = px/px.shift(60)-1
vol20 = rets.rolling(20).std()
f1 = (mom20 - mom60)/vol20
# factor 2: dn_mkt_beta_60d = beta on down-market days
mkt_neg = rets.mean(axis=1).clip(upper=0)
f2 = rets.rolling(60).cov(mkt_neg)/mkt_neg.rolling(60).var()
# factor 3: rate_beta_cn10y_60d = beta of asset ret on CN10Y change
cn10y = px['CN10Y']
dcn = cn10y.pct_change()
f3 = rets.rolling(60).cov(dcn)/dcn.rolling(60).var()

print("\n=== factor rank-IC (h=10) trailing windows ===")
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    r = rank_ic(f, fwd)
    print(name)
    for look,(ic, sd, n) in r.items():
        icir = ic/sd if sd and sd>0 else np.nan
        print(f"  {look}d: IC={ic:.4f} ICIR={icir:.3f} n={n}")

# latest factor exposures (last cross-section)
print("\n=== latest cross-section factor values ===")
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    last = f.iloc[-1].dropna()
    print(name, "valid:", len(last), "| top:", last.nlargest(4).index.tolist(), "| bottom:", last.nsmallest(4).index.tolist())

# dispersion trend: std of 20d returns across assets rolling
disp = rets.iloc[-120:].std(axis=1)
print("\ncross-sectional dispersion (std of daily returns across 15): last 20d avg:", round(disp.iloc[-20:].mean()*100,2),
      " prior 60d avg:", round(disp.iloc[-80:-20].mean()*100,2), " last:", round(disp.iloc[-1]*100,2))
