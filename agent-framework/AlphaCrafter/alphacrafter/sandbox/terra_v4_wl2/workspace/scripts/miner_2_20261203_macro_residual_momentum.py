import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-12-03')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index();P[s]=d[d.index<=cutoff]
P=pd.DataFrame(P).sort_index();R=P.pct_change();m=R.mean(1)
beta=R.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
r5=P.pct_change(5);res=r5-beta.mul(m.rolling(5).sum(),axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
z=(v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std(); f=res.mul(np.where(z.values[:,None]<0,1,-1))
fr=R.shift(-1); rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8:rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',len(a)/(len(P)-1));print('IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean()));print('years',a.groupby(a.index.year).ic.mean().round(5).to_dict());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
