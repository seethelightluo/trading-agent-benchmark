import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=get_account_dict()['watch_list']; frames={}
for s in U:
 try: d=get_index_daily_data(s, days=4000)
 except Exception: d=None
 if d is None:
  try: d=get_stock_daily_data(s, days=4000)
  except Exception: d=None
 if d is not None and len(d): frames[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill(); ret=px.pct_change(); r5=px.pct_change(5); csres=r5.sub(r5.median(axis=1),axis=0); vol=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
factor=(-csres/vol).replace([np.inf,-np.inf],np.nan).shift(1); fwd=px.shift(-10)/px-1; rows=[]
for dt in factor.index:
 a=factor.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  c=a[ok].corr(b[ok])
  if np.isfinite(c): rows.append((dt,c,int(ok.sum())))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for window in [None,120,252]:
 q=z if window is None else z.tail(window); mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
 print('window',window,'dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/len(U),'IC %.8f ICIR %.8f hit %.4f'% (mean,icir,(q.ic>0).mean()),flush=True)
print('decay',flush=True)
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; rr=[]
 for dt in factor.index:
  a=factor.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   c=a[ok].corr(b[ok])
   if np.isfinite(c): rr.append(c)
 print(h, np.nanmean(rr),len(rr),flush=True)
rank=factor.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'assets',len(frames),'dates',len(px),px.index.min(),px.index.max(),flush=True)
print('blocks',flush=True)
for inds in np.array_split(np.arange(len(z)),4):
 q=z.iloc[inds]; print(len(q),q.ic.mean(),q.ic.std(ddof=1),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),flush=True)
