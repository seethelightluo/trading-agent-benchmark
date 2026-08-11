import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}; P=pd.DataFrame(px).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(20); rel=R.sub(R.median(axis=1),axis=0)
z=(v-v.rolling(120,min_periods=60).mean())/v.rolling(120,min_periods=60).std(); reg=1/(1+np.exp(z.clip(-4,4))); f=rel.mul(reg,axis=0)
for h in [5,10,20]:
 a=[]
 for dt in P.loc[:cut].index:
  fut=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a);print(h,len(a),len(a)/len(P.loc[:cut].index),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
valid=f.loc[:cut].notna().sum(axis=1); print('coverage',valid.mean()/15,'avg',valid.mean(),'turnover',f.loc[:cut].rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'corr_raw',f.stack().corr(R.stack()))
# annual 10d
out=[]
for dt in P.loc[:cut].index:
 q=pd.concat([f.loc[dt],(P.shift(-10).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
 if len(q)>=8:out.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
d=pd.DataFrame(out,columns=['date','ic']).set_index('date');print(d.groupby(d.index.year).ic.agg(['count','mean']).to_string())
