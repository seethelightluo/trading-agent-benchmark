import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2035-01-31']
r=P.pct_change(); r20=P.pct_change(20); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Relative reversal: fade each asset's 20d return relative to contemporaneous cross-section;
# scale by its own risk and use breadth to avoid overreacting in uniformly trending markets.
rel=r20.sub(r20.mean(axis=1),axis=0)
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1)
gate=(0.55+0.9*(breadth-0.5)).clip(0.2,1.1)
F=(-rel/vol).mul(gate,axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']); s=out.ic
print('data_last',P.index.max().date(),'period',out.date.min().date(),out.date.max().date(),'dates',len(out),'avgN',round(out.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True)
out.to_csv('scripts/artifacts/miner_2_20350201_breadth_relative_reversal_20d_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_2_20350201_breadth_relative_reversal_20d_signal.csv')
