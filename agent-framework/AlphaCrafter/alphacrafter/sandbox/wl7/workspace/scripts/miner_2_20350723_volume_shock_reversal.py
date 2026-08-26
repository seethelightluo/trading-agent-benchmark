import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def main():
 P={}
 for s in U:
  d=None
  try:d=get_index_daily_data(s,4200)
  except:pass
  if d is None:
   try:d=get_stock_daily_data(s,4200)
   except:pass
  if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close
 p=pd.DataFrame(P).sort_index(); r=p.pct_change(); r5=p.pct_change(5)
 rv20=r.rolling(20,min_periods=15).std(); vr=pd.DataFrame()
 # volume is not consistently available; use range shock as an equivalent observable liquidity/attention proxy
 # factor: reverse 5d move, amplified by abnormal true range, risk normalized
 high={}; low={}
 for s in U:
  d=None
  try:d=get_index_daily_data(s,4200)
  except:pass
  if d is None:
   try:d=get_stock_daily_data(s,4200)
   except:pass
  if d is not None: high[s]=d.set_index(pd.to_datetime(d.date)).high; low[s]=d.set_index(pd.to_datetime(d.date)).low
 h=pd.DataFrame(high).reindex(p.index); l=pd.DataFrame(low).reindex(p.index)
 range20=((h-l)/p).rolling(20,min_periods=15).mean(); base=range20.rolling(60,min_periods=30).median()
 shock=(range20/(base+1e-12)).clip(.25,4)
 fac=(-r5/(rv20*np.sqrt(5)+1e-12))*shock
 fac=fac.shift(1)
 f=p.shift(-20).div(p)-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): vals.append(c);ns.append(len(z))
 a=np.array(vals); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 for n in [252,756,1260]:
  q=a[-n:];print('recent',n,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
 ranks=fac.rank(axis=1,pct=True); top=ranks>=.8; changes=[]
 for i in range(1,len(top)):
  a1=top.iloc[i-1];a2=top.iloc[i];z=a1.notna()&a2.notna()
  if z.sum()>=8: changes.append((a1[z]!=a2[z]).mean())
 print('coverage',fac.notna().mean().mean(),'turnover',np.mean(changes))
 fac.to_csv('scripts/miner_2_20350723_volume_shock_reversal_signal.csv',index_label='date')
if __name__=='__main__':main()
