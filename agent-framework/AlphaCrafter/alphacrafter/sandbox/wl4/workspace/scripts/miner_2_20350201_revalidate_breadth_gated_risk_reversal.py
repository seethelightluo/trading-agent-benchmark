import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2035-01-31']
r=P.pct_change(); trend=P.pct_change(60); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1); gate=(0.5+0.8*(breadth-0.5)).clip(.25,1.05)
F=(-trend/vol).mul(gate,axis=0).shift(1); rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']); s=o.ic
print('period',o.date.min().date(),o.date.max().date(),'dates',len(o),'avgN',round(o.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
o.to_csv('scripts/artifacts/miner_2_20350201_breadth_gated_risk_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20350201_breadth_gated_risk_reversal_signal.csv')
