import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Macro-conditioned medium momentum: stress proxy reverses the momentum orientation only in an extreme rising-volatility regime.
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv')
 if len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
V=pd.read_csv('../persistent/index_data/VIX.csv')
V=V.set_index(pd.to_datetime(V.date)).close.astype(float)
df=pd.concat(px,axis=1).sort_index(); ret=df.pct_change()
vix=V.reindex(df.index).ffill()
# common dates and forward return
rows={k:[] for k in ['base','conditioned','conditioned_soft']}; nins=[]; dates=[]
for i in range(60,len(df)-10):
 dt=df.index[i]
 vals=ret.iloc[i-20:i].add(1).prod()-1
 vol=ret.iloc[i-20:i].std()*np.sqrt(252)
 base=vals/vol.replace(0,np.nan)
 # VIX level above 75th rolling percentile and rising over 5d: reversal regime
 hist=vix.iloc[max(0,i-252):i].dropna()
 stress=(len(hist)>100 and vix.iloc[i-5:i].mean()>vix.iloc[i-10:i-5].mean() and vix.iloc[i]>hist.quantile(.75))
 cond=-base if stress else base
 soft=base*(0.35 if stress else 1.0)
 fwd=df.iloc[i+10]/df.iloc[i]-1
 ok=base.notna()&fwd.notna()
 if ok.sum()>=8:
  dates.append(dt); nins.append(ok.sum())
  for k,x in [('base',base),('conditioned',cond),('conditioned_soft',soft)]: rows[k].append(x[ok].corr(fwd[ok]))
print('dates',len(dates),'avg_instruments',np.mean(nins),'coverage',np.mean(nins)/15)
for k,x in rows.items():
 a=np.asarray(x); print(k,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'std',np.nanstd(a,ddof=1))
 print('regimes',[(y,round(np.nanmean([a[j] for j,q in enumerate(dates) if pd.Timestamp(q).year==y]),5),sum(pd.Timestamp(q).year==y for q in dates)) for y in range(2020,2031) if sum(pd.Timestamp(q).year==y for q in dates)>0])
# decay for selected
for h in [5,10,20]:
 z=[]
 for i in range(60,len(df)-h):
  vals=ret.iloc[i-20:i].add(1).prod()-1; vol=ret.iloc[i-20:i].std()*np.sqrt(252); f=vals/vol.replace(0,np.nan)
  hist=vix.iloc[max(0,i-252):i].dropna(); stress=len(hist)>100 and vix.iloc[i-5:i].mean()>vix.iloc[i-10:i-5].mean() and vix.iloc[i]>hist.quantile(.75)
  if stress:f=-f
  fr=df.iloc[i+h]/df.iloc[i]-1; ok=f.notna()&fr.notna()
  if ok.sum()>=8:z.append(f[ok].corr(fr[ok]))
 print('decay',h,np.nanmean(z))
