import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
# Candidate: short-horizon residual shock reversal, activated only when cross-asset dispersion is elevated.
# All components are lagged one completed day at decision time.
raw=-(0.7*r.rolling(3,min_periods=3).sum()+0.3*r.rolling(10,min_periods=10).sum())/(r.rolling(30,min_periods=20).std()*np.sqrt(5)+1e-12)
disp=r.rolling(20,min_periods=15).std().mean(axis=1).shift(1)
thresh=disp.rolling(120,min_periods=60).median().shift(1)
gate=(disp>thresh).astype(float)
sig=raw.shift(1).mul(gate,axis=0)
# cross-sectional rank is the tested signal; gated dates remain valid with zero signals
sig=sig.rank(axis=1,pct=True).sub(.5)
y1=P.shift(-1)/P-1

def evalh(y):
 vals=[]; ns=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'));ns.append(v.sum())
 a=pd.Series(vals); return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns),a
n,ic,ir,hit,av,a=evalh(y1)
print('rows',len(P),'assets',len(P.columns),'dates',n,'avg_n %.2f'%av)
print('daily IC %.8f ICIR %.8f hit %.5f'%(ic,ir,hit))
for h in [5,10,20]:
 n2,i2,ir2,_,_,_=evalh(P.shift(-h)/P-1); print('h',h,'dates',n2,'IC %.8f ICIR %.8f'%(i2,ir2))
print('coverage %.5f turnover %.5f gated_share %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),gate.mean()))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
rows=[]
for dt,x in zip(sig.index[a.index if False else []],[]): pass
pd.DataFrame({'ic':a}).to_csv('scripts/miner_3_20310505_dispersion_shock_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310505_dispersion_shock_reversal_signal.csv',index=False)
