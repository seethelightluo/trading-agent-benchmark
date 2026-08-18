import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2035-02-14']
r=P.pct_change(); v=r.rolling(20,min_periods=15).std(); cs=r.rolling(5,min_periods=5).sum(); inv=(1/v).clip(upper=inv if False else np.inf)
# Relative, inverse-volatility scaled short-term strength, lagged one day.
F=cs.div(v).sub(cs.div(v).mean(axis=1),axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']); s=o.ic
print('period',o.date.min().date(),o.date.max().date(),'dates',len(o),'avgN',round(o.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); o.to_csv('scripts/artifacts/miner_2_20350215_inversevol_relative_strength_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20350215_inversevol_relative_strength_signal.csv')
