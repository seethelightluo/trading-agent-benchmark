import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}; P=pd.DataFrame(P).sort_index().loc[:'2034-05-02']; R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill().pct_change()
# Asset-specific rolling VIX beta; penalize VIX-sensitive names in reversal score, using only lagged data.
beta=R.rolling(120,min_periods=60).cov(v).div(v.rolling(120,min_periods=60).var(),axis=0)
beta_z=beta.sub(beta.mean(axis=1),axis=0).div(beta.std(axis=1),axis=0).clip(-2,2)
r20=P.pct_change(20); med=r20.median(axis=1); vol=R.rolling(40,min_periods=25).std()*np.sqrt(252)
F=(-(r20.sub(med,axis=0)).div(vol) * (1-0.25*beta_z)).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_3_20340511_vix_beta_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20340511_vix_beta_reversal_signal.csv')
