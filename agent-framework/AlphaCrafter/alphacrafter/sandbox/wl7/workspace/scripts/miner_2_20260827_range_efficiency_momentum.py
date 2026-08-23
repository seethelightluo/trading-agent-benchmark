import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 try:d=get_index_daily_data(s,3000)
 except Exception:d=None
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except Exception:d=None
 if d is None or len(d)==0:return None
 d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None};print('assets',len(D))
rows=[]
for s,d in D.items():
 c=d.close.replace([np.inf,-np.inf],np.nan); r=c.pct_change()
 # trend efficiency: directional 20d move relative to path length, scaled by volatility
 f=(c/c.shift(20)-1)/(r.abs().rolling(20).sum()+1e-12)
 # lag signal one day to avoid same close
 f=f.shift(1); fr=c.shift(-1)/c-1
 z=pd.DataFrame({'f':f,'fr':fr,'asset':s}).dropna().reset_index();rows.append(z)
R=pd.concat(rows,ignore_index=True);obs=[]
for dt,g in R.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:obs.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date');print('obs',len(I),'avg_n',I.n.mean());
def stat(a):return (a.mean(),a.std(ddof=1),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean())
print('daily mean std ICIR hit',stat(I.ic))
for h in [5,10,20]:
 rr=[]
 for s,d in D.items():
  c=d.close;r=c.pct_change();f=((c/c.shift(20)-1)/(r.abs().rolling(20).sum()+1e-12)).shift(1);fr=c.shift(-h)/c-1
  rr.append(pd.DataFrame({'f':f,'fr':fr}).dropna().assign(asset=s).reset_index())
 q=pd.concat(rr);a=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 print('h',h,'obs',len(a),'stats',stat(pd.Series(a)))
p=R.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean(axis=1).mean(),'coverage',len(R)/sum(len(d) for d in D.values()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b,'ic'];print('regime',a,b,'obs',len(z),'stats',stat(z) if len(z)>1 else None)
print('max abs library corr unavailable; signal artifact not persisted')
