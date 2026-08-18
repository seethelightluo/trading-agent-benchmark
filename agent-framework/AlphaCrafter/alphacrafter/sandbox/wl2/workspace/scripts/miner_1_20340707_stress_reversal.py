import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date'); px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
stress=(vix/vix.rolling(60,min_periods=30).median()-1).clip(lower=0)
sig=-(p/p.shift(3)-1).mul((1+stress),axis=0)
sig=sig.clip(lower=sig.quantile(.05,axis=1),upper=sig.quantile(.95,axis=1),axis=0)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([sig.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=stress_conditioned_3d_reversal','dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
for lab,x in [('all',r),('2026+',r[r.index>='2026']),('2030+',r[r.index>='2030']),('2032+',r[r.index>='2032'])]: print(lab,'IC %.6f ICIR %.6f hit %.4f dates %d'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(),len(x)))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
