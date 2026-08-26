import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-12-26']; R=P.pct_change()
# Tail-resilience reversal: fade 40d losses, but penalize assets whose downside volatility is elevated.
r40=P.pct_change(40); down=R.where(R<0).rolling(60,min_periods=30).std(); total=R.rolling(60,min_periods=30).std()
resilience=(r40/(down*np.sqrt(60)+1e-8))
# cross-sectional rank, with low downside-risk quality tilt; winsorize robustly
F=-resilience
F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('candidate=tail_resilience_reversal; universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
 x=np.array(vals,float); ok=np.isfinite(x); x=x[ok]; ds=pd.DatetimeIndex(dates)[ok]; ns=np.array(ns)[ok]; sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',round(ns.mean(),2),'coverage',round(ns.mean()/len(U),4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round(np.mean(x>0),4))
 for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2029-12-26'),('recent','2028-12-27','2029-12-26')]:
  z=x[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]; zs=z.std(ddof=1)
  print(' ',label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/zs,6) if len(z)>1 and zs else None,'hit',round(np.mean(z>0),4) if len(z) else None)
rank=F.rank(axis=1,pct=True); print('turnover_rank_abs_change',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'valid_coverage',round(F.notna().mean().mean(),6))
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20291227_tail_resilience_40d_signal.csv',index=False)
