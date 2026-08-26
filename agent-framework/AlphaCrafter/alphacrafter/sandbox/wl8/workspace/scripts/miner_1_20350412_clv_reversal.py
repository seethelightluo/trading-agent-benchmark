import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); ds[s]=d
ix=pd.Index(sorted(set().union(*[set(x.index) for x in ds.values()])))
# 5-day average close-location value, then invert: buying weak closes should revert
sig=pd.DataFrame(index=ix,columns=U,dtype=float); close=pd.DataFrame(index=ix,columns=U,dtype=float)
for s,d in ds.items():
 c=d.close.reindex(ix); hi=d.high.reindex(ix); lo=d.low.reindex(ix)
 cl=((c-lo)/(hi-lo).replace(0,np.nan)-.5).rolling(5,min_periods=3).mean()
 sig[s]=-cl; close[s]=c
fwd=close.shift(-10)/close-1; out=[]; sr=[]
for dt in ix:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 for s in U:
  if pd.notna(sig.loc[dt,s]): sr.append((dt,s,float(sig.loc[dt,s])))
o=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('dates',len(o),'avg_n',o.n.mean(),'coverage',len(sr)/(len(ix)*15));print('ic',o.ic.mean(),'icir',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'turnover',sig.rank(pct=True).diff().abs().stack().mean())
for w in [365,750,1260]:
 q=o.tail(w);print('window',w,'ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(),'n',len(q))
for h in [1,5,10,20]:
 yy=close.shift(-h)/close; a=[]
 for dt in ix:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(a),len(a))
o.to_csv('scripts/miner_1_20350412_clv_reversal_ic.csv');pd.DataFrame(sr,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20350412_clv_reversal_signal.csv',index=False)
