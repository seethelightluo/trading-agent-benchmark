import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-11-18'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); short=r.rolling(10,min_periods=8).std(); long=r.rolling(60,min_periods=30).std()
# Breakout continuation: recent 5d direction weighted by volatility compression, cross-sectionally winsorized.
f=(P.pct_change(5)/(short+1e-8)*(long/(short+1e-8))).replace([np.inf,-np.inf],np.nan)
f=f.clip(lower=f.quantile(.1,axis=1),upper=f.quantile(.9,axis=1),axis=0)
def run(h,idx=f.index):
 y=P.pct_change(h).shift(-h);v=[];n=[]
 for d in idx:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:v.append(spearmanr(z.f,z.y).statistic);n.append(len(z))
 q=np.array(v);return q,n
q,n=run(1);print('all dates',len(q),'avgN',round(np.mean(n),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2026-11')]:
 q,_=run(1,f.loc[a:b].index);print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for h in [3,5,10]:q,_=run(h);print('decay',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4));f.to_csv('scripts/miner_1_20261119_compression_breakout_signal.csv')
