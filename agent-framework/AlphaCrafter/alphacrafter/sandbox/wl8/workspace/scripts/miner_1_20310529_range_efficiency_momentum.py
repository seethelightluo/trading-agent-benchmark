import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,days=3000)
    except Exception: pass
    if d is None:
        try: d=get_stock_daily_data(s,days=3000)
        except Exception: pass
    if d is not None and len(d)>100:
        raw[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close']
idx=sorted(set.intersection(*[set(v.index) for v in raw.values()]))
P=pd.DataFrame({s:raw[s].reindex(idx) for s in raw}).sort_index(); R=P.pct_change()
# Prior 20-day directional return divided by realized absolute daily path; shift avoids current-day data.
f=(P.pct_change(20).shift(1))/(R.abs().rolling(20).sum().shift(1))
fwd=P.shift(-10)/P-1
rows=[]; sig=[]
for dt in P.index:
 a,b=f.loc[dt],fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(b[ok],method='spearman'),int(ok.sum())))
  sig.append(a.rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(sig,index=q.index)
for label,z in [('all',q),('recent365',q.tail(365)),('recent180',q.tail(180)),('recent60',q.tail(60))]:
 m=z.ic.mean(); sd=z.ic.std(ddof=1); print(label,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),6) if sd else np.nan,'hit',round((z.ic>0).mean(),4))
print('coverage',round(q.n.sum()/(len(q)*len(U)),4),'turnover',round(sr.diff().abs().mean().mean(),6),'assets',len(raw),'dates',len(q),'period',q.index.min(),q.index.max())
f.to_csv('scripts/miner_1_20310529_range_efficiency_momentum_signal.csv')
q.to_csv('scripts/miner_1_20310529_range_efficiency_momentum_ic.csv')
