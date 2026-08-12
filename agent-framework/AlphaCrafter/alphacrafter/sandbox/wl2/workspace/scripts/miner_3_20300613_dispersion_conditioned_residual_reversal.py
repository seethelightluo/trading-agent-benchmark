import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None and len(d)>100: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# High-dispersion conditioned residual reversal. Dispersion is known at t,
# signal uses completed t bar and predicts t+1. Residual removes common cross-asset move.
csmean=R.mean(axis=1); residual=R.sub(csmean,axis=0)
disp=R.std(axis=1,ddof=1); gate=disp>disp.rolling(60,min_periods=30).median()
vol=R.rolling(20,min_periods=15).std()
f=(-residual/(vol+1e-8)).where(gate,0.0)
rows=[]; sig=[]
for i,t in enumerate(P.index[:-10]):
 x=f.loc[t].replace([np.inf,-np.inf],np.nan)
 for h in (1,5,10):
  y=P.shift(-h).loc[t]/P.loc[t]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: rows.append((t,h,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
 sig.append(x.rename(t))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('H',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20300613_dispersion_conditioned_residual_reversal_signal.csv',index_label='date')
print('assets',len(px),'signal_rows',len(S),'dates',len(P))
