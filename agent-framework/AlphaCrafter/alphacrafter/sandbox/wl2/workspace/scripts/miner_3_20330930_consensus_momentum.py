import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=6000)
            if x is not None and len(x)>100: return x
        except Exception: pass
    return None
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
print('assets',len(raw),{s:len(x) for s,x in raw.items()})
# aligned close panel, factor uses completed t and is evaluated t+10 forward
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index()
r=np.log(p).diff()
# cross-sectional percentile ranks at each date, two horizon consensus, lagged one day
m20=p.pct_change(20); m60=p.pct_change(60)
f=(m20.rank(axis=1,pct=True)+m60.rank(axis=1,pct=True))/2
f=f.shift(1)
rows=[]
for h in [1,3,5,10,20]:
  fr=p.shift(-h)/p-1
  vals=[]
  for d in f.index:
    a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
    if len(a)>=8: vals.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
  z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
  print('H',h,'dates',len(z),'avgN',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'cov',len(z)/len(f))
  for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
    zz=z.loc[lo:hi]
    if len(zz): print(' ',lo,hi,len(zz),round(zz.ic.mean(),5),round(zz.ic.mean()/zz.ic.std(ddof=1),4))
  if h==10:
    z.to_csv('scripts/miner_3_20330930_consensus_momentum_10d_ic.csv')
# turnover: rank signal changes
print('turnover', (f.diff().abs().mean(axis=1).mean()), 'valid signal',f.notna().mean().mean())
