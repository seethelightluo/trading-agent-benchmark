import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-06-13')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; P[s]=d[d.index<=cut]
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); f=((p/p.shift(20)-1)-(p/p.shift(60)-1)/3)/r.rolling(20,min_periods=15).std(); f=f.clip(-8,8)
def calc(h):
 a=[]; ds=[]; ns=[]
 for i in range(len(p)-h):
  if p.index[i]<pd.Timestamp('2026-01-01') or p.index[i+h]>cut: continue
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.f,z.y).statistic); ds.append(p.index[i]); ns.append(len(z))
 return np.array(a),np.array(ds),ns
x,d,n=calc(10); print('full',len(x),d[0],d[-1],np.mean(n),np.mean(n)/15,x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
for k in [180,360]:
 y=x[-k:];print('recent',k,y.mean(),y.mean()/y.std(ddof=1))
for h in [5,20]:
 y,_,_=calc(h);print('decay',h,y.mean(),y.mean()/y.std(ddof=1),len(y))
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean();print('turn',turn)
pd.DataFrame({'date':d,'ic':x}).to_csv('scripts/miner_1_20300613_acceleration_ic.csv',index=False)
f.loc[d].stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300613_acceleration_signal.csv',index=False)
