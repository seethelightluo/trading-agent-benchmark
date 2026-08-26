import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy();d.date=pd.to_datetime(d.date);cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index();r=P.pct_change();m=r.mean(axis=1)
mm=m.rolling(60,min_periods=40).mean(); vv=((m-mm)**2).rolling(60,min_periods=40).mean()+1e-12
beta=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for c in r:
 xm=r[c].rolling(60,min_periods=40).mean(); beta[c]=((r[c]-xm)*(m-mm)).rolling(60,min_periods=40).mean()/vv
res=r-beta.mul(m,axis=0)
raw=res.rolling(20,min_periods=15).sum()/(res.rolling(30,min_periods=20).std()+1e-12)
weak=(m.rolling(20,min_periods=15).sum()<0).astype(float)
sig=raw.shift(1).mul(weak.shift(1),axis=0).rank(axis=1,pct=True).sub(.5)
def ev(h):
 y=P.shift(-h)/P-1;vals=[];ns=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8: vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'));ns.append(v.sum())
 a=pd.Series(vals);return a,ns
a,ns=ev(1);print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean(ns));print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 z,_=ev(h);print('h',h,'dates',len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.5f turnover %.5f weak_share %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),weak.mean()));print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
a.to_csv('scripts/miner_1_20310519_beta_neutral_weak_momentum_ic.csv',header=['ic']);sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310519_beta_neutral_weak_momentum_signal.csv',index=False)
