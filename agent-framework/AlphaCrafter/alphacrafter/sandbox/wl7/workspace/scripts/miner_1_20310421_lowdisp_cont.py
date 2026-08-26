import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=5000)
  except Exception: pass
  if d is not None: break
 if d is not None and len(d)>=300: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); vol=r.rolling(20).std()
# Low-dispersion continuation: normalized 5-day relative return, activated only when
# cross-asset dispersion is unusually compressed (lagged percentile; no look-ahead).
cs=r.rolling(20).std().mean(axis=1)
low=cs < cs.rolling(252,min_periods=126).quantile(.25).shift(1)
rel=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
sig=rel/vol*low.astype(float).values[:,None]
y=P.shift(-1).div(P).sub(1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=q.ic.mean(); icir=ic/q.ic.std(ddof=1)*np.sqrt(252)
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('rows',len(P),'assets',len(px),'dates',len(q),'avg_n',q.n.mean())
print('IC %.9f ICIR %.9f hit %.4f coverage %.4f turnover %.5f'%(ic,icir,(q.ic>0).mean(),q.n.mean()/len(U),turn))
for h in [5,10,20]:
 yy=P.shift(-h).div(P).sub(1); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
for label,sub in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]: print(label,'ic',sub.ic.mean(),'n',len(sub))
print('period',P.index.min(),P.index.max())
sig.to_csv('scripts/miner_1_20310421_lowdisp_cont_signal.csv',index_label='date'); q.to_csv('scripts/miner_1_20310421_lowdisp_cont_ic.csv')
