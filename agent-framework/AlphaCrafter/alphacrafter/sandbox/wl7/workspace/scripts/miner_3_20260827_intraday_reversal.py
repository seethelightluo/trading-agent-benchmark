import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    try: d=get_index_daily_data(s,3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is None or len(d)==0: return None
    d=d.copy(); d['date']=pd.to_datetime(d['date']).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
print('assets',len(D), 'ranges', {s:(str(d.index.min().date()),str(d.index.max().date()),len(d)) for s,d in D.items()})
rows=[]
for s,d in D.items():
    x=d[['open','close']].replace([np.inf,-np.inf],np.nan).dropna()
    x['f']=-(x['close']/x['open']-1.0); x['fr']=x['close'].shift(-1)/x['close']-1.0; x=x.dropna(); x['asset']=s; rows.append(x[['f','fr','asset']].reset_index())
R=pd.concat(rows,ignore_index=True); ics=[]
for dt,g in R.groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: ics.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date'); print('obs',len(I),'avg_n',I.n.mean()); print('IC mean/std/IR/hit',I.ic.mean(),I.ic.std(ddof=1),I.ic.mean()/I.ic.std(ddof=1)*np.sqrt(252),(I.ic>0).mean())
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items():
  f=-(d.close/d.open-1); x=d.close.shift(-h)/d.close-1; z=pd.concat([f,x],axis=1).dropna(); z.columns=['f','fr']; z['asset']=s; rr.append(z.reset_index())
 q=pd.concat(rr); ii=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: ii.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(ii).dropna(); print('h',h,'n',len(a),'ic',a.mean(),'ir',a.mean()/a.std(ddof=1)*np.sqrt(252))
p=R.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b,'ic']; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
