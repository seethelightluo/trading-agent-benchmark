import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-11-28'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change()
# Defensive anchor: equal-weight XAU, US10Y and CN10Y; rank reversal is stronger after a drawdown followed by positive recovery.
defn=p[['XAU','US10Y','CN10Y']].pct_change(10).mean(axis=1)
rel=p.pct_change(10).sub(defn,axis=0)
dd=p/p.rolling(60,min_periods=40).max()-1
rec=p.pct_change(5)
# continuous interpretable score: contrarian relative return, amplified only for assets recovering from a material drawdown
amp=1+2*((dd<-0.05)&(rec>0)).astype(float)
s=(-rel*amp).replace([np.inf,-np.inf],np.nan)
def run(h):
 xs=[];ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));ds.append(p.index[i])
 x=pd.Series(xs,index=ds);return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(np.array(ns)/15)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:
 z=run(h);print('TEST',h,'dates',z[0],'IC',round(z[1],6),'ICIR',round(z[2],6),'hit',round(z[3],4),'coverage',round(z[4],4))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291129_defensive_relative_recovery_signal.csv',index=False)
print('artifact_rows',len(out),'turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-11-28')]:
 xs=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'))
 q=pd.Series(xs);print('REG10',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('max_abs_library_correlation',None)
