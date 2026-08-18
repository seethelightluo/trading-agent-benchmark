import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); prices[a]=d.close.astype(float)
P=pd.DataFrame(prices).sort_index().loc[:'2033-07-20']; R=P.pct_change(); ret30=P.pct_change(30)
resid=ret30.sub(ret30.mean(axis=1),axis=0)
cons=R.rolling(60,min_periods=40).apply(lambda x: np.mean(x>0),raw=True)
F=(-resid*(0.5+cons)).shift(1); rows=[]
for dt in F.index:
 fut=P.shift(-30).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('cutoff',P.index.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20330721_consistency_residual_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20330721_consistency_residual_reversal_signal.csv')
