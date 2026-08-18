import pandas as pd,numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2035-09-27'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]; p[s]=d
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); rv=r.rolling(20).std(); f=-(rv.rolling(60).mean()/rv.rolling(60).std())
# use valid >=8 and 10d forward returns
vals=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],p.pct_change(10).shift(-10).iloc[i]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
vals=np.array(vals)
print('cutoff',cut.date(),'dates',len(vals),'avg instruments',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean())
print('IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1),'hit',np.mean(vals>0),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n,a,b in [('early',0,len(vals)//3),('middle',len(vals)//3,2*len(vals)//3),('recent',2*len(vals)//3,len(vals))]: print(n,vals[a:b].mean(),len(vals[a:b]))
f.to_csv('scripts/miner_2_20350928_stable_lowvol_signal.csv',index_label='date')
