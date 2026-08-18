import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d[d.date<=pd.Timestamp('2034-09-29')].set_index('date').sort_index(); px[a]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change(); base=prices.pct_change(60).shift(1); vol=ret.rolling(20).std().shift(1).clip(lower=0.002)
breadth=(base>0).mean(axis=1); mult=np.where(breadth<.4,.5,1.)
scaled=base/vol; sig=-scaled.sub(scaled.median(axis=1),axis=0).mul(mult,axis=0); sig=sig.clip(sig.quantile(.05,axis=1),sig.quantile(.95,axis=1),axis=0)
rows=[]
for i in range(len(prices)-10):
 z=pd.concat([sig.iloc[i].rename('s'),(prices.iloc[i+10]/prices.iloc[i]-1).rename('f')],axis=1).dropna()
 if len(z)>=8: rows.append((prices.index[i],spearmanr(z.s,z.f).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ranks=sig.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()*2
print('assets',len(px),'dates',len(r),'start',r.index.min(),'end',r.index.max(),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*len(px)))
print('IC',r.ic.mean(),'ICIR_daily',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turnover',turnover)
for label,sub in [('2024-27',r.loc['2024-01-01':'2027-12-31']),('2028-31',r.loc['2028-01-01':'2031-12-31']),('2032+',r.loc['2032-01-01':])]: print(label,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1),(sub.ic>0).mean())
out=pd.DataFrame(sig.stack(),columns=['signal']); out.index.names=['date','asset']; out.to_csv('../persistent/miner_2_20340929_contrarian_mom60_volnorm_signal.csv')
