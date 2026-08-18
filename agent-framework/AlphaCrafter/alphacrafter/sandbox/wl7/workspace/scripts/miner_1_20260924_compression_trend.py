import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    try: d=get_index_daily_data(s,3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is None or len(d)==0:return None
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 r=d.close.pct_change(); v20=r.rolling(20).std(); v60=r.rolling(60).std()
 # compression-adjusted trend: recent risk-adjusted momentum rewarded when short vol is below long vol
 f=r.rolling(5).sum()/(v20*np.sqrt(5)+1e-12)*(v60/(v20+1e-12)).clip(0.5,2.0)
 # signal is known after close; forward next-session return
 z=pd.DataFrame({'f':f,'fr':d.close.shift(-1)/d.close-1}).dropna();z['asset']=s;rows.append(z.reset_index())
R=pd.concat(rows,ignore_index=True); obs=[]
for dt,g in R.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:obs.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date');
print('assets',len(D),'obs',len(I),'avg_n',I.n.mean(),'coverage',len(I)/(len(pd.concat([d for d in D.values()]).index.unique())))
print('daily mean std ICIR hit',I.ic.mean(),I.ic.std(ddof=1),I.ic.mean()/I.ic.std(ddof=1)*np.sqrt(252),(I.ic>0).mean())
for h in [1,5,10]:
 rr=[]
 for s,d in D.items():
  r=d.close.pct_change();v20=r.rolling(20).std();v60=r.rolling(60).std();f=r.rolling(5).sum()/(v20*np.sqrt(5)+1e-12)*(v60/(v20+1e-12)).clip(.5,2);x=pd.DataFrame({'f':f,'fr':d.close.shift(-h)/d.close-1}).dropna();rr.append(x.assign(asset=s).reset_index())
 q=pd.concat(rr);a=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a).dropna();print('h',h,'obs',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252))
p=R.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b,'ic'];print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
