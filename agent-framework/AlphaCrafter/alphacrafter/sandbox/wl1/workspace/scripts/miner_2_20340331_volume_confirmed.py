import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p={}; vol={}
for s in U:
 f=os.path.join(b,s+'.csv')
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date');p[s]=d.close;vol[s]=d.volume
P=pd.DataFrame(p).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
R=P.pct_change(); market=R.mean(axis=1)
# volume-confirmed medium momentum: return times log volume impulse, cross-sectional z implicit via rank
mom=P.pct_change(20); vi=np.log1p(V).diff(20); raw=mom*(1+vi.clip(-2,2)/2)
# risk adjust and lag
sig=(raw/R.rolling(20).std()).shift(1); y=P.shift(-10)/P-1
rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'avgN',a.n.mean(),'coverage',a.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean()))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=a.loc[lo:hi].ic;print(lo,hi,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; rr=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.mean(rr),'n',len(rr))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20340331_volume_confirmed_signal.csv',index=False)
