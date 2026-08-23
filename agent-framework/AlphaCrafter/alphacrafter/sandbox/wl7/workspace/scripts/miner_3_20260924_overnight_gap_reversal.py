import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,3000)
            if d is not None and len(d):
                d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
        except Exception: pass
    return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
    x=d[['open','close']].replace([np.inf,-np.inf],np.nan).dropna()
    # prior close to open gap; negative gap receives positive reversal score
    x['f']=-(x.open/x.close.shift(1)-1)
    x['fr']=x.close.shift(-1)/x.close-1
    x=x.dropna(); x['asset']=s; rows.append(x[['f','fr','asset']].reset_index())
R=pd.concat(rows,ignore_index=True); out=[]
for dt,g in R.groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: out.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); sd=I.ic.std(ddof=1)
print('assets',len(D),'dates',len(I),'avg_instruments',I.n.mean(),'period',I.index.min().date(),I.index.max().date())
print('daily_ic',I.ic.mean(),'daily_icir',I.ic.mean()/sd*np.sqrt(252),'hit', (I.ic>0).mean())
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items():
  f=-(d.open/d.close.shift(1)-1); fr=d.close.shift(-h)/d.close-1
  z=pd.concat([f,fr],axis=1).dropna(); z.columns=['f','fr']; rr.append(z.assign(asset=s).reset_index())
 q=pd.concat(rr); ii=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: ii.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(ii).dropna(); print('horizon',h,'dates',len(a),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1)*np.sqrt(252))
p=R.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).mean(),'coverage',len(I)/len(pd.date_range(I.index.min(),I.index.max())))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b,'ic']; print('regime',a,b,'dates',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(252))
