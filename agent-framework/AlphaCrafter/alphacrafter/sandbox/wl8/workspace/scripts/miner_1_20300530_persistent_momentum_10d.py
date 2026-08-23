import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-05-30')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Interpretable trend persistence: intermediate return, scaled by fraction of up days.
r10=p.pct_change(10); up=r.rolling(10,min_periods=8).apply(lambda x: np.mean(x>0),raw=True)
vol=r.rolling(20,min_periods=15).std()
fac=(r10/vol)*(0.5+up)
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(len(p)-10):
 dt=p.index[i]; end=p.index[i+10]
 if end>cut or dt<p.index[25]: continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v): ics.append(v); dates.append(dt); ns.append(ok.sum())
 cov.append(ok.mean())
 if i:
  a=x.rank(pct=True); b=fac.iloc[i-1].rank(pct=True); oo=a.notna()&b.notna()
  if oo.sum(): turns.append(np.abs(a[oo]-b[oo]).mean())
ics=np.array(ics); dates=np.array(dates,dtype='datetime64[ns]'); m=ics.mean(); sd=ics.std(ddof=1)
print({'factor':'persistent_volnorm_momentum_10d','dates':len(ics),'start':str(dates[0]),'end':str(dates[-1]),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(m),'icir':float(m/sd),'hit':float(np.mean(ics>0)),'turnover':float(np.mean(turns))})
for name,mask in [('recent180',dates>=np.datetime64('2029-12-01')),('recent360',dates>=np.datetime64('2029-01-01')),('2028',(dates>=np.datetime64('2028-01-01'))&(dates<np.datetime64('2029-01-01'))),('2029',(dates>=np.datetime64('2029-01-01'))&(dates<np.datetime64('2030-01-01'))),('2030',dates>=np.datetime64('2030-01-01'))]:
 z=ics[mask]; print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_1_20300530_persistent_momentum_10d_ic.csv',index=False)
# signal artifact: latest factor cross section
fac.iloc[-1].rename('signal').to_csv('scripts/miner_1_20300530_persistent_momentum_10d_signal.csv')
