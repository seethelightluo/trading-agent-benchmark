import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None and len(d): return d
        except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 down=r.where(r<0,0.0)
 dvol=np.sqrt((down.pow(2).rolling(40).mean()))
 # risk-adjusted medium trend, penalizing downside shocks rather than upside variability
 f=(r.rolling(20).sum())/(dvol*np.sqrt(20)+1e-12)
 S[s]=pd.DataFrame({'f':f,'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})

def calc(col):
 rows=[]
 dates=sorted(set().union(*[x.index for x in S.values()]))
 for dt in dates:
  vals=[x.loc[dt] for x in S.values() if dt in x.index and np.isfinite(x.loc[dt,['f',col]]).all()]
  if len(vals)>=8:
   z=pd.DataFrame(vals); rows.append((dt,z.f.rank().corr(z[col].rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n'])
 m=q.ic.mean(); sd=q.ic.std(ddof=1)
 return q,m,m/sd*np.sqrt(252), (q.ic>0).mean()
for col,h in [('f1',1),('f5',5),('f10',10)]:
 q,m,ir,hit=calc(col)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(m,6),'ICIR',round(ir,4),'hit',round(hit,4))
 if h==10:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
   v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a+'-'+b,len(v),round(v.mean(),6) if len(v) else None)
  # signal turnover proxy: rank top/bottom changes on consecutive common dates
  print('valid_assets',len(S))
