import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-08-29')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None};
# Idiosyncratic momentum: 30-session asset return residualized against equal-weight universe return,
# scaled by residual volatility; all rolling inputs lagged before the forecast.
P=pd.concat({s:pd.to_numeric(d.close,errors='coerce') for s,d in D.items()},axis=1).sort_index(); R=P.pct_change()
M=R.mean(axis=1); rows=[]
for s in D:
 r=R[s]; cov=r.rolling(60,min_periods=45).cov(M); var=M.rolling(60,min_periods=45).var(); beta=cov/var
 resid=r-beta*M; rv=resid.rolling(30,min_periods=20).std()
 f=((P[s].pct_change(30)-beta.shift(1)*M.rolling(30,min_periods=20).sum())/(rv*np.sqrt(252)+0.01)).shift(1)
 x=pd.DataFrame({'date':P.index,'asset':s,'f':f,'fr1':P[s].shift(-1)/P[s]-1,'fr5':P[s].shift(-5)/P[s]-1,'fr10':P[s].shift(-10)/P[s]-1,'fr20':P[s].shift(-20)/P[s]-1})
 rows.append(x)
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def st(x,col):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1: z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for x in ['fr1','fr5','fr10','fr20']: print(x,st(q,x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr10'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
q.to_csv('scripts/miner_1_20270818_idio_momentum_signal.csv',index=False)
