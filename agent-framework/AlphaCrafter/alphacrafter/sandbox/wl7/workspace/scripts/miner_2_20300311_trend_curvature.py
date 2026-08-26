import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: d=get_index_daily_data(s,days=4000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,days=4000)
        except Exception: d=None
    if d is not None: D[s]=d
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items()}).sort_index().ffill()
r=np.log(px).diff(); vol=r.rolling(20,min_periods=15).std()
ret20=np.log(px/px.shift(20)); ret60=np.log(px/px.shift(60))
# Curvature: recent trend minus half medium trend, scaled by recent volatility; lag one day.
sig=(ret20-0.5*ret60)/vol
sig=sig.rank(axis=1,pct=True)-0.5
sig=sig.shift(1)
rows=[]
for t in sig.index:
    fwd=np.log(px.shift(-10).loc[t]/px.loc[t]); z=pd.concat([sig.loc[t],fwd],axis=1).dropna()
    if len(z)>=8: rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'dates',len(out),'avg_n',out.n.mean(),'coverage',sig.notna().mean().mean(),'ic',out.ic.mean(),'icir',out.ic.mean()/out.ic.std(),'hit',(out.ic>0).mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-03-10')]:
 q=out.loc[a:b].ic; print('regime',a,b,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std() if len(q)>1 else np.nan,'hit',(q>0).mean())
for h in [1,5,10,20,40]:
 rr=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t],np.log(px.shift(-h).loc[t]/px.loc[t])],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'dates',len(rr),'icir',np.nanmean(rr)/np.nanstd(rr)*np.sqrt(len(rr)))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20300311_trend_curvature_signal.csv',index=False)
out.to_csv('scripts/miner_2_20300311_trend_curvature_ic.csv')
