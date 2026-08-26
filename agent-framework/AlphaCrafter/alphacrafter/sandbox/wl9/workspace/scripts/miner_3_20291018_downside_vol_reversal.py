import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-10-17']; R=P.pct_change(); look=20
down=R.clip(upper=0).rolling(look,min_periods=15).std(); ret=P.pct_change(look)
F=-(ret/(down*np.sqrt(252)+1e-12)); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('candidate=downside_vol_scaled_reversal; universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
 x=np.array(vals,float); ok=np.isfinite(x); x=x[ok]; ds=pd.DatetimeIndex(dates)[ok]; sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',np.mean(ns),'coverage',np.mean(ns)/len(U),'IC',x.mean(),'ICIR',x.mean()/sd,'hit',np.mean(x>0),'sd',sd)
 for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2029-10-17'),('recent','2028-10-18','2029-10-17')]:
  z=x[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]; zs=z.std(ddof=1)
  print(' ',label,'dates',len(z),'IC',z.mean() if len(z) else None,'ICIR',z.mean()/zs if len(z)>1 and zs else None)
rank=F.rank(axis=1,pct=True); print('turnover_rank_abs_change',rank.diff().abs().mean(axis=1).dropna().mean(),'valid_coverage',F.notna().mean().mean())
