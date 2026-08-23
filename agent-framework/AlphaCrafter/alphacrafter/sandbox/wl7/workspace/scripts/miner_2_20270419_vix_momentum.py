import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-18')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
p=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']).dt.normalize(); v=v.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
vc=v['close'].astype(float).reindex(p.index).ffill(); state=(vc<vc.rolling(60,min_periods=30).median()).astype(float); # calm=1, stressed=0
rows=[]
for s in D:
 rr=r[s]; mom=rr.rolling(20).sum().shift(1); vol=rr.rolling(40).std().shift(1)*np.sqrt(20)
 # retain momentum in calm regime; attenuate/reverse it during stress
 f=(mom/(vol+1e-12)*(0.25+0.75*state)).shift(1)
 rows.append(pd.DataFrame({'date':p.index,'asset':s,'f':f}))
base=pd.concat(rows,ignore_index=True)
def run(h, subset=None):
 xs=[]
 for s in D:
  z=base[base.asset.eq(s)].copy(); z['fr']=p[s].shift(-h).reindex(z.date).to_numpy()/p[s].reindex(z.date).to_numpy()-1; xs.append(z)
 x=pd.concat(xs).replace([np.inf,-np.inf],np.nan).dropna()
 if subset is not None:x=x[subset.reindex(x.date).fillna(False).to_numpy()]
 vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(vals);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
q=base.dropna(); print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*15),'vix rows',vc.notna().sum())
for h in [1,5,10,20]:print('horizon',h,run(h))
for label,mask in [('calm',state>0.5),('stress',state<=0.5)]:print(label,run(1,mask))
rank=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean().mean());q.to_csv('scripts/miner_2_20270419_vix_momentum_signal.csv',index=False)
