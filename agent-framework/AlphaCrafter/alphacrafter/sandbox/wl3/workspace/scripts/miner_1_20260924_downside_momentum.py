import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:'2026-09-23']
r=P.pct_change()
# downside-risk-adjusted medium momentum: 20d return penalized by recent negative-return volatility
neg=r.where(r<0)
down=neg.rolling(30,min_periods=15).std()
f=P.pct_change(20)/(down*np.sqrt(20)+1e-8)
# winsorize cross-section to limit crypto outliers
f=f.clip(lower=f.quantile(.1,axis=1),upper=f.quantile(.9,axis=1),axis=0)
for h in [1,3,5,10]:
 y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in f.index:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 x=np.array(vals);print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2026-09')]:
 y=P.pct_change().shift(-1); vals=[]
 for d in f.loc[a:b].index:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8:vals.append(spearmanr(z.f,z.y).statistic)
 x=np.array(vals);print('regime',a,b,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20260924_downside_momentum_signal.csv')
