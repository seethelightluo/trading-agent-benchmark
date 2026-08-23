import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    try: d=get_index_daily_data(s,days=3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,days=3000)
        except Exception: d=None
    if d is not None and len(d)>100:
        x=d[['date','close']].copy(); raw[s]=x.set_index('date')
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()]))
P=pd.DataFrame({s:raw[s].reindex(idx).close for s in raw}).sort_index(); R=P.pct_change()
ret20=P.pct_change(20).shift(1); path=R.abs().rolling(20).sum().shift(1); f=ret20/path
fwd=P.shift(-10)/P-1
rows=[]; signals=[]
for dt in P.index:
 a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum())); signals.append(a.rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(signals,index=q.index)
turn=sr.diff().abs().mean().mean()
for label,z in [('all',q),('recent365',q.tail(365)),('recent180',q.tail(180)),('recent60',q.tail(60))]:
 mean=z.ic.mean(); sd=z.ic.std(ddof=1); print(label,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(mean,6),'ICIR',round(mean/sd*np.sqrt(252),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(sum(q.n)/(len(q)*len(U)),4),'turnover',round(float(turn),6),'assets',len(raw),'dates',len(q),'period',q.index.min(),q.index.max())
f.to_csv('scripts/miner_3_20310515_range_efficiency_momentum_signal.csv')
