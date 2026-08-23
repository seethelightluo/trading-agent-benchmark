import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-06-06']; R=P.pct_change()
neg2=(R.clip(upper=0)**2); d20=np.sqrt(neg2.rolling(20,min_periods=15).mean()); d60=np.sqrt(neg2.rolling(60,min_periods=40).mean())
F=(-(d20/d60)).shift(1).replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
z=pd.DataFrame(rows,columns=['date','n','ic']); s=z['ic']
print('period',z['date'].min().date(),z['date'].max().date(),'dates',len(z),'avgN',round(z.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); z.to_csv('scripts/artifacts/miner_1_20350607_downside_vol_compression_revalidation_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20350607_downside_vol_compression_revalidation_signal.csv')
