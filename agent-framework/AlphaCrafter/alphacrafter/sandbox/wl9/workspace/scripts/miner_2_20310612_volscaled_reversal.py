import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>150:D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}); r=c.pct_change()
# Contrarian version of volatility-scaled medium-term trend; signal is lagged one session.
trend=r.rolling(60,min_periods=40).sum()/(r.rolling(40,min_periods=25).std()*np.sqrt(252)+1e-12)
sig=(-trend).sub((-trend).median(axis=1),axis=0).shift(1)
for h in [5,10,20,40]:
 a=[]
 for i in range(len(c)-h):
  z=pd.DataFrame({'s':sig.iloc[i],'f':c.iloc[i+h]/c.iloc[i]-1}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append((c.index[i],z.s.corr(z.f),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for n,a1,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   x=q[(q.date>=a1)&(q.date<=b)]; print('REG',n,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6) if len(x)>1 else np.nan,round((x.ic>0).mean(),4) if len(x) else np.nan)
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20310612_volscaled_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
