import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2030-01-09']; R=P.pct_change(); r20=P.pct_change(20); down=R.clip(upper=0).fillna(0).rolling(30,min_periods=20).std(); disp=R.rolling(20,min_periods=15).std().mean(axis=1); med=disp.rolling(120,min_periods=60).median(); gate=(disp/(med+1e-8)).clip(.5,2.); F=(-(r20/(down+1e-8))).mul(gate,axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
fr=P.shift(-5).div(P)-1; vals=[];dates=[];ns=[]
for dt in F.index:
 a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c); dates.append(dt); ns.append(len(a))
x=np.array(vals); ds=pd.DatetimeIndex(dates); sd=x.std(ddof=1)
print('universe',len(U),'dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/len(U),4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round(np.mean(x>0),4))
for lab,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2030-01-09'),('recent','2029-01-10','2030-01-09')]:
 z=x[(ds>=lo)&(ds<=hi)]; print(lab,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'valid_coverage',round(F.notna().mean().mean(),6))
pd.DataFrame({'date':np.repeat(F.index,len(U)),'symbol':U*len(F.index),'signal':F.to_numpy().ravel()}).dropna().to_csv('scripts/miner_2_20300110_dispersion_gated_downside_reversal_5d_signal.csv',index=False)
