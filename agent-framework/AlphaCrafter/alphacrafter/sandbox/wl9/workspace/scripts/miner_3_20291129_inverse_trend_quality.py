import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-11-28']; R=P.pct_change(); ret=P.pct_change(60); vol=R.rolling(20,min_periods=15).std()*np.sqrt(252); path=R.abs().rolling(60,min_periods=45).sum(); eff=ret.abs()/(path+1e-12)
# Contrarian trend-quality: reverse the previously tested continuation signal.
F=-(ret/(vol+1e-12))*eff
F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('candidate=inverse_trend_quality; universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); dates.append(dt); ns.append(len(a))
 x=np.asarray(vals); ds=pd.DatetimeIndex(dates); sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/len(U),4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round(np.mean(x>0),4))
 for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2029-11-28'),('recent','2028-11-29','2029-11-28')]:
  z=x[(ds>=lo)&(ds<=hi)]; zs=z.std(ddof=1); print(' ',label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/zs,6) if len(z)>1 and zs>0 else None,'hit',round(np.mean(z>0),4) if len(z) else None)
rank=F.rank(axis=1,pct=True); print('turnover_rank_abs_change',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'valid_coverage',round(F.notna().mean().mean(),6))
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20291129_inverse_trend_quality_10d_signal.csv',index=False)
