import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}; V={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date'); C[a]=d.close.astype(float); V[a]=d.volume.astype(float)
P=pd.DataFrame(C).sort_index().loc[:'2034-01-25']; Vol=pd.DataFrame(V).reindex(P.index).sort_index(); R=P.pct_change(); rv=R.rolling(20,min_periods=15).std()
# Reversal strengthened when recent move occurs on unusually high volume; lagged one day.
shock=(Vol/Vol.rolling(60,min_periods=40).median()).replace([np.inf,-np.inf],np.nan)
F=(-P.pct_change(20).div(rv)*shock).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20340202_volume_confirmed_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20340202_volume_confirmed_reversal_signal.csv')