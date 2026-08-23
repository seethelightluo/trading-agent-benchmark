import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-05-16')
px={}
for s in U:
    f='../persistent/stock_data/'+s+'.csv'
    d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
    px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index()
# one candidate: consistency-adjusted 20d momentum, rewarding positive daily-return persistence
rets=p.pct_change()
mean20=rets.rolling(20,min_periods=15).mean()
vol20=rets.rolling(20,min_periods=15).std()
r20=p.pct_change(20)
# blend trend magnitude and consistency; cross-sectional ranking is used by portfolio
fac=(r20/vol20)*((mean20>0).astype(float)*0.5+0.5)
ics=[]; turns=[]; cov=[]; ninst=[]; dates=[]
for i in range(len(p)-10):
    dt=p.index[i]; end=p.index[i+10]
    if end>cut or dt<p.index[25]: continue
    x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1
    ok=x.notna()&y.notna()
    if ok.sum()<8: continue
    ic=spearmanr(x[ok],y[ok]).statistic
    if np.isfinite(ic):
      ics.append(ic); ninst.append(ok.sum()); dates.append(dt)
    # turnover of rank signal at observed daily dates
    if i>0:
      z=x.rank(pct=True); q=fac.iloc[i-1].rank(pct=True)
      oo=z.notna()&q.notna()
      if oo.sum(): turns.append((z[oo]-q[oo]).abs().mean())
    cov.append(ok.mean())
ics=np.array(ics); m=ics.mean(); sd=ics.std(ddof=1); ir=m/sd if sd else np.nan
print({'factor':'consistency_adjusted_momentum_20d','dates':len(ics),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_instruments':float(np.mean(ninst)),'coverage':float(np.mean(cov)),'ic':float(m),'icir':float(ir),'hit':float((ics>0).mean()),'turnover':float(np.mean(turns)),'ic5_proxy':None})
for name,mask in [('recent180',np.array(dates)>=pd.Timestamp('2029-11-18')),('recent360',np.array(dates)>=pd.Timestamp('2028-12-01')),('2028', (np.array(dates)>=pd.Timestamp('2028-01-01'))&(np.array(dates)<pd.Timestamp('2029-01-01'))),('2029', (np.array(dates)>=pd.Timestamp('2029-01-01'))&(np.array(dates)<pd.Timestamp('2030-01-01'))),('2030',np.array(dates)>=pd.Timestamp('2030-01-01'))]:
 z=ics[mask]; print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
# save artifact for provenance
out=pd.DataFrame({'date':dates,'ic':ics}); out.to_csv('scripts/miner_3_20300516_consistency_momentum_10d_signal.csv',index=False)
