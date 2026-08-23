import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-03-02')
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
D={s:load(s) for s in U}; v= pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']).dt.normalize(); v=v.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
# Candidate: volatility-regime-conditioned medium-horizon reversal. Signal is fully lagged.
vp=v.close.astype(float); vz=((vp-vp.rolling(60,min_periods=30).mean())/(vp.rolling(60,min_periods=30).std()+1e-12)).shift(1)
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
 f=(-(c.pct_change(10))/(vol+1e-12)).where(vz.reindex(c.index)>0.75)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':c.shift(-1)/c-1}))
q=pd.concat(rows).replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('candidate=VIX-conditioned 10d volatility-scaled reversal')
print('dates',q.date.nunique(),'assets',len(U),'avg_instruments',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(U)))
for h in [1,5,10,20]:
 if h==1: x=q
 else:
  z=[]
  for s,d in D.items():
   f=q[q.asset==s].set_index('date').f.reindex(d.index); z.append(pd.DataFrame({'date':d.index,'asset':s,'f':f.values,'fr':(d.close.shift(-h)/d.close-1).values}))
  x=pd.concat(z).replace([np.inf,-np.inf],np.nan).dropna()
 print('h',h,'n_dates avg_n IC ICIR hit',stats(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('rank_turnover',float(r.diff().abs().mean().mean()))
q.to_csv('scripts/miner_3_20270302_vix_conditioned_reversal_signal.csv',index=False)
