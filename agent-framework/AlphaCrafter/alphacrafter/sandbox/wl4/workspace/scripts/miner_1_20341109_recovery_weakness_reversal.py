import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2034-11-08']
r=P.pct_change(); r60=P/P.shift(60)-1; rollmax=P.rolling(60,min_periods=45).max(); mdd=(1-P/rollmax).rolling(60,min_periods=45).max()
# Contrarian recovery weakness: assets with poor 60d return per unit trailing drawdown.
F=(-(r60/(mdd+0.05))).replace([np.inf,-np.inf],np.nan).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
R=pd.DataFrame(rows,columns=['date','n','ic']); s=R.ic
print('period',R.date.min().date(),R.date.max().date(),'dates',len(R),'avgN',round(R.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); R.to_csv('scripts/artifacts/miner_1_20341109_recovery_weakness_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20341109_recovery_weakness_reversal_signal.csv')
