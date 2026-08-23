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
    if d is not None and len(d)>100: raw[s]=d[['date','close']].set_index('date')
idx=sorted(set.intersection(*[set(x.index) for x in raw.values()]))
P=pd.DataFrame({s:raw[s].reindex(idx).close for s in raw}).sort_index(); R=P.pct_change()
# Lagged 20d return divided by recent downside deviation; rewards upside with penalized downside risk.
down=R.where(R<0,0).pow(2).rolling(20).mean().pow(.5).shift(1)
f=P.pct_change(20).shift(1)/down.replace(0,np.nan)
fwd=P.shift(-10)/P-1
rows=[]; sig=[]
for dt in P.index:
 a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(b[ok],method='spearman'),ok.sum())); sig.append(a.rank(pct=True))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sr=pd.DataFrame(sig,index=q.index)
for label,z in [('all',q),('recent365',q.tail(365)),('recent180',q.tail(180)),('recent60',q.tail(60))]:
 m=z.ic.mean(); sd=z.ic.std(ddof=1); print(label,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(m,6),'ICIR_daily',round(m/sd,6),'ICIR_ann',round(m/sd*np.sqrt(252),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(sum(q.n)/(len(q)*len(U)),4),'turnover',round(float(sr.diff().abs().mean().mean()),6),'assets',len(raw),'dates',len(q),'period',q.index.min(),q.index.max())
f.to_csv('scripts/miner_3_20310529_downside_adjusted_momentum_signal.csv')
q.to_csv('scripts/miner_3_20310529_downside_adjusted_momentum_ic.csv')
