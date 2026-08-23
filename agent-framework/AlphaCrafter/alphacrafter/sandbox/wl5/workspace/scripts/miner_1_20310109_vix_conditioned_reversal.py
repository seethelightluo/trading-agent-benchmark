import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); med=R.median(axis=1); resid=R.sub(med,axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(P.index).ffill()
# Candidate: VIX-regime conditioned 20-session residual reversal, only when VIX is elevated versus its causal 120d history.
def factor(i):
 shock=resid.iloc[i-19:i+1].sum(); vol=resid.iloc[i-59:i+1].std()*np.sqrt(20)
 pct=(v.iloc[i-1] >= v.iloc[max(0,i-120):i].quantile(.70))
 return -shock/(vol+1e-12)*float(pct)
def run(h):
 rows=[]; signals=[]
 for i in range(130,len(P)-h):
  f=factor(i); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  for s,a in f.items(): signals.append((P.index[i],s,float(a)))
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); ir=m/x.ic.std(ddof=1)
 print('horizon',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(ir,6),'hit',round((x.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-01-08')]:
  q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else np.nan)
 if h==20:
  S=pd.DataFrame(signals,columns=['date','symbol','signal']).pivot(index='date',columns='symbol',values='signal')
  print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
  pd.DataFrame(signals,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20310109_vix_conditioned_reversal_signal.csv',index=False)
for h in [5,10,20]: run(h)
