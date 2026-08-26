import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}; vol={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); px[s]=d.close.astype(float); vol[s]=d.volume.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2030-01-23']; V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change()
# Volume-shock reversal: fade recent losses when accompanied by unusually high turnover.
r5=P.pct_change(5); vz=V/(V.rolling(60,min_periods=20).median()+1e-12)-1
# cross-sectional residual removes common market move; high volume amplifies the contrarian shock
res=r5.sub(r5.median(axis=1),axis=0)
F=(-res)*(1+vz.clip(lower=0,upper=3)); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('candidate=volume_shock_reversal; universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
 x=np.array(vals,float); ok=np.isfinite(x); x=x[ok]; ds=pd.DatetimeIndex(dates)[ok]; ns=np.array(ns)[ok]; sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',round(ns.mean(),2),'coverage',round(ns.mean()/len(U),4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round(np.mean(x>0),4))
 for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2030-01-23'),('recent','2029-01-24','2030-01-23')]:
  z=x[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]; zs=z.std(ddof=1)
  print(' ',label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/zs,6) if len(z)>1 and zs else None,'hit',round(np.mean(z>0),4) if len(z) else None)
rank=F.rank(axis=1,pct=True); print('turnover_rank_abs_change',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'valid_coverage',round(F.notna().mean().mean(),6))
F.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300124_volume_shock_reversal_10d_signal.csv',index=False)
