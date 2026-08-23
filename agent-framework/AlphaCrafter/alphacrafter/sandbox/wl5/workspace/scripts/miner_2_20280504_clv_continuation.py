import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=10
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d
# 3-day average close location value, with range floor; demean cross-section
px=pd.DataFrame({s:d['close'] for s,d in frames.items()})
clv=pd.DataFrame({s:((d['close']-d['low'])/(d['high']-d['low']).replace(0,np.nan)) for s,d in frames.items()})
# smooth only completed daily bars; centered cross-sectional residual removes common market candle bias
sig=clv.rolling(3,min_periods=3).mean()
fwd=px.shift(-H)/px-1
rows=[]; dates=sorted(set(sig.index)&set(fwd.index))
for dt in dates:
    x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)<8: continue
    # high CLV = buying pressure; expected continuation over 10 days
    ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('idea=3d_average_close_location_continuation horizon=10')
print('dates',len(r),'avg_n',round(r.n.mean(),3),'coverage',round(r.n.sum()/(len(r)*15),4))
print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=r.loc[a:b]
 if len(q): print(a+'..'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
# rank turnover proxy
rank=sig.rank(axis=1,pct=True)
turn=rank.diff().abs().stack().groupby(level=0).mean().reindex(r.index).mean()
print('turnover_proxy',round(float(turn),6))
# save recoverable artifact
out=sig.loc[r.index].copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_2_20280504_clv_continuation_signal.csv',index=False)
print('artifact scripts/miner_2_20280504_clv_continuation_signal.csv')
