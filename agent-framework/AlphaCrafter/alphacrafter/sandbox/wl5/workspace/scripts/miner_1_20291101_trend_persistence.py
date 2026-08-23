import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-10-31')
p={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}; p=pd.DataFrame(p).sort_index().ffill(); r=p.pct_change()
# Causal trend persistence: 20d return, risk-normalized, requiring positive 5d confirmation.
s=(p.pct_change(20)/(r.rolling(40,min_periods=30).std()*np.sqrt(20)+1e-12))*((p.pct_change(5)>0)*.5+.5); s=s.rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 v=[];ns=[]
 for i in range(260,len(p)-h):
  d=p.index[i]
  if a and not(pd.Timestamp(a)<=d<=pd.Timestamp(b)):continue
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:v.append(q.f.corr(q.y,method='spearman'));ns.append(len(q))
 x=np.array(v);return len(x),np.mean(ns)/15,np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [5,10,20]:print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-10-31')]:print('REG10',a,b,calc(10,a,b))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20291101_trend_persistence_signal.csv',index=False);print('artifact',len(out),out.date.nunique(),out.symbol.nunique(),'coverage',s.notna().mean().mean(),'turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean())
