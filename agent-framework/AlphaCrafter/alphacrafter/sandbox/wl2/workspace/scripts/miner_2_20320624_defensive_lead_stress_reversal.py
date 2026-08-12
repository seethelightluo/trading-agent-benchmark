import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 d=pd.read_csv('../persistent/stock_data/'+sym+'.csv',parse_dates=['date']);d.date=d.date.dt.normalize();return d.set_index('date').sort_index()
D={s:load(s) for s in U};dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<='2032-06-23')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change();V=R.rolling(20,min_periods=15).std()
defensive=R[['XAU','US10Y','CN10Y']].rolling(5,min_periods=5).sum().mean(axis=1); rel=R.rolling(5,min_periods=5).sum().sub(defensive,axis=0)
disp=R.T.rolling(3,min_periods=3).std().T.mean(axis=1); threshold=disp.rolling(252,min_periods=100).quantile(.60); breadth=(R.rolling(3,min_periods=3).sum()>0).mean(axis=1);stress=(disp>=threshold)&(breadth<=.45)
F=(-rel.div(V)).where(stress,np.nan).shift(1);Y=C.shift(-1).div(C)-1
def calc(yy):
  a=[];ns=[];ds=[]
  for dt in dates:
   z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();z=z[z.f!=0]
   if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
  return np.array(a),ns,ds
a,ns,ds=calc(Y);print('candidate defensive_lead_stress_reversal_5d');print('dates',len(a),'avgN',round(np.mean(ns),2),'active_dates',len(a),'coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4));print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lo,hi in [(2020,2022),(2023,2025),(2026,2029),(2030,2032)]:
 q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>2 else 'insufficient')
for h in [3,5,10]:
 q,_,_=calc(C.shift(-h).div(C)-1);print('decay',h,'IC %.6f'%q.mean())
out=F.copy();out.index.name='date';out.to_csv('scripts/miner_2_20320624_defensive_lead_stress_reversal_signal.csv')
