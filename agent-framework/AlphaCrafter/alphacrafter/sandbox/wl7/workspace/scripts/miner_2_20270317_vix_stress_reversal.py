import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: VIX-stress-conditioned lagged 5d reversal, normalized by realized vol.
frames={}
for s in U:
 d=get_stock_daily_data(s, days=3000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').close
  frames[s]=d
px=pd.DataFrame(frames).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
# signal at t uses through t-1; VIX z uses through t-1
ret=px.pct_change(); vol=ret.rolling(20).std()
base=-(px.pct_change(5).shift(1))/vol.shift(1)
z=(v.rolling(60).mean().shift(1)-v.rolling(252).mean().shift(1))/v.rolling(252).std().shift(1)
# bounded stress multiplier, neutral at normal stress
mult=(1+0.35*np.tanh(z)).reindex(px.index).fillna(1.0)
f=base.mul(mult,axis=0)
ics=[]; turns=[]; ninst=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(px)-h):
  x=f.iloc[i]; y=px.pct_change(h).iloc[i+h]
  q=pd.concat([x,y],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1])); ninst.append(len(q))
 a=np.array(vals); print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ninst[-len(a):]),2),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std()*np.sqrt(252),6),'hit',round(np.mean(a>0),4))
# turnover and coverage
r=f.rank(axis=1,pct=True); turns=np.nanmean(np.abs(r.diff()).mean(axis=1)); cov=f.notna().sum(axis=1).mean()/15
print('coverage',round(cov,6),'rank_turnover',round(turns,6),'dates',len(f),'assets',len(frames))
# regime split daily
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 x=[]
 for i in range(len(px)-1):
  if str(px.index[i].year)[:4] not in []:
   if a<=str(px.index[i].year)<=b:
    q=pd.concat([f.iloc[i],px.pct_change().iloc[i+1]],axis=1).dropna()
    if len(q)>=8:x.append(q.iloc[:,0].corr(q.iloc[:,1]))
 print('regime',a,b,'n',len(x),'IC',round(np.mean(x),7) if x else None)
# artifact for provenance
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20270317_vix_stress_reversal_signal.csv',index=False)
