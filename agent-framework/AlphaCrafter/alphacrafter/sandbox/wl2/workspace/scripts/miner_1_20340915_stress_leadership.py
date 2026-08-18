import pandas as pd, numpy as np, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+a+'.csv'
 px[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
w=pd.DataFrame(px).sort_index().ffill().loc[:'2034-09-14']
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(w.index).ffill()
vz=((v-v.rolling(60,min_periods=30).mean())/(v.rolling(60,min_periods=30).std()+1e-12)).clip(-1,2)
base=w.pct_change(20).sub(w.pct_change(20).median(axis=1),axis=0)
f=(base*(1+0.7*vz.clip(lower=0)).to_numpy()[:,None]).shift(1)
for h in [1,3,5,10,20]:
 fr=w.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(vals).dropna(); print(h,'dates',len(s),'meanIC',s.mean(),'ICIR',s.mean()/(s.std(ddof=1)+1e-12),'hit',(s>0).mean())
valid=f.notna().sum(1)/15; ranks=f.rank(axis=1,pct=True)
print('coverage',valid.mean(),'avg_n',f.notna().sum(1).mean(),'turnover',ranks.diff().abs().mean(axis=1).dropna().mean(),'period',w.index.min(),w.index.max())
for label,mask in [('2026-28',(f.index>='2026-07-16')&(f.index<'2029-01-01')),('2029-31',(f.index>='2029-01-01')&(f.index<'2032-01-01')),('2032-34',(f.index>='2032-01-01'))]:
 fr=w.pct_change(10).shift(-10); q=[]
 for dt in f.index[mask]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(label,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12))
f.to_csv('scripts/miner_1_20340915_stress_leadership_signal.csv')
