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
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
vol=r.rolling(20).std()
shock=-(0.7*r.rolling(3).sum()/vol - 0.3*r.rolling(10).sum()/vol)
disp=r.std(axis=1).rolling(20).mean(); th=disp.rolling(252,min_periods=126).quantile(.75).shift(1)
sig=shock.mul((disp>th).astype(float),axis=0)
fwd=P.shift(-1).div(P).sub(1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=q.ic.mean(); sd=q.ic.std(ddof=1); icir=ic/sd*np.sqrt(252)
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('rows',len(P),'assets',len(px),'dates',len(q),'avg_n',q.n.mean())
print('IC %.9f ICIR %.9f hit %.4f coverage %.4f turnover %.5f'%(ic,icir,(q.ic>0).mean(),q.n.mean()/len(U),turn))
for h in [5,10,20]:
 yy=P.shift(-h).div(P).sub(1); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
sig.to_csv('scripts/miner_1_20310224_disp75_blend_signal.csv',index_label='date'); q.to_csv('scripts/miner_1_20310224_disp75_blend_ic.csv')
