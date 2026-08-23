import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1900)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); ret=p.pct_change()
# medium-term momentum normalized by realized volatility, with confirmation by sign breadth
mom=p/p.shift(60)-1
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
bread=(ret.rolling(20,min_periods=15).mean()>0).astype(float).rolling(10,min_periods=5).mean()
f=(mom/(vol+1e-6))*(0.5+0.5*bread)
# signal available after close t; evaluate next close-to-close return
f=f.shift(1); fr=ret
rows=[]; vals=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.x.corr(z.y),len(z)))
  vals.append(z.x.corr(z.y))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=r.ic.mean(); sd=r.ic.std(ddof=1); icir=mean/sd*np.sqrt(len(r)) if sd else np.nan
# turnover rank changes on overlapping valid signals
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
print({'dates':len(r),'avg_n':r.n.mean(),'ic':mean,'icir':icir,'hit':(r.ic>0).mean(),'coverage':f.notna().sum().sum()/(f.shape[0]*len(U)),'turnover':turnover})
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h)
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.x.corr(z.y))
 print('h',h,'ic',np.nanmean(a),'n',len(a))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=r.loc[lo:hi].ic.dropna(); print(lo,{'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan})
# signal artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20270203_vol_scaled_60d_momentum_signal.csv',index=False)
