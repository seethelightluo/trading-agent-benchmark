import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_index_daily_data(s,3000)
 except:d=None
 if d is None:
  try:d=get_stock_daily_data(s,3000)
  except:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
# each asset's rolling 5d return, then subtract same-day cross-sectional median
M=pd.DataFrame({s:d.close.pct_change(5) for s,d in D.items()}); med=M.median(axis=1); F=M.sub(med,axis=0)
rows=[]
for s,d in D.items():
 z=pd.DataFrame({'f':F[s],'fr':d.close.shift(-1)/d.close-1}).dropna();rows.append(z.assign(asset=s).reset_index())
R=pd.concat(rows,ignore_index=True);out=[]
for dt,g in R.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:out.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');print('assets',len(D),'obs',len(I),'avg_n',I.n.mean(),'IC',I.ic.mean(),'std',I.ic.std(),'ICIR',I.ic.mean()/I.ic.std(ddof=1)*np.sqrt(252),'hit',(I.ic>0).mean())
for h in [5,10]:
 rr=[]
 for s,d in D.items():
  z=pd.DataFrame({'f':F[s],'fr':d.close.shift(-h)/d.close-1}).dropna();rr.append(z.assign(asset=s).reset_index())
 q=pd.concat(rr);a=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a);print('h',h,'obs',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b].ic;print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
