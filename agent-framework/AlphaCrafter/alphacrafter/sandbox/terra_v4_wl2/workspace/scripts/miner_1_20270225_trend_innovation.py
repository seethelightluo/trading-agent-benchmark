import pandas as pd,numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A if os.path.exists('../persistent/stock_data/'+a+'.csv')}
px=pd.DataFrame(p).sort_index();r=px.pct_change();
# orthogonal innovation: short return minus its own medium average, volatility scaled
f=((r.rolling(5,min_periods=5).sum()-r.rolling(20,min_periods=20).sum()/4)/(r.rolling(20,min_periods=20).std()+1e-8)).shift(1)
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]; ranks=[]
 for d in f.index:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);dates.append(d);ns.append(len(z));ranks.append(z.f.rank(pct=True))
 q=np.array(vals);print('H',h,'dates',len(q),'avg_n',np.mean(ns),'coverage',np.mean(f.notna().sum(axis=1)/len(A)),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0),'turnover',pd.DataFrame(ranks).diff().abs().mean().mean())
 for name,mask in [('2020-22',[d.year<=2022 for d in dates]),('2023-24',[2023<=d.year<=2024 for d in dates]),('2025-26',[2025<=d.year<=2026 for d in dates]),('2027',[d.year==2027 for d in dates])]:
  z=q[mask];print(name,len(z),z.mean() if len(z) else np.nan,z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
f.to_csv('../persistent/factor_signals_miner_1_20270225_trend_innovation.csv')
