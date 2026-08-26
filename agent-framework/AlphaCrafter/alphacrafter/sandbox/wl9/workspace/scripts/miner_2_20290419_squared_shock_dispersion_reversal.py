import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None and len(x): return x
  except: pass
D={s:fetch(s) for s in U}; P=pd.DataFrame({s:x.set_index('date').close for s,x in D.items() if x is not None}).sort_index().ffill(); R=P.pct_change(); disp=R.std(axis=1); gate=disp>disp.rolling(60).median()
# nonlinear shock reversal: stronger reversal signal for larger prior-day shocks, activated in dispersed sessions
F=(-R*np.abs(R)).where(gate,np.nan)
rows=[]
for h in [1,5,10]:
 FW=P.shift(-h)/P-1; out=[]
 for d in P.index:
  z=pd.concat([F.loc[d],FW.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(out,columns=['d','ic','n']).set_index('d'); ic=q.ic.dropna(); print('H',h,'dates',len(ic),'assets',q.n.mean(),'coverage',q.n.sum()/(len(ic)*len(U)),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
 for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31'),('2028-04-05','2029-04-18')]:
  x=q.loc[a:b].ic.dropna(); print(a,len(x),round(x.mean(),5),round(x.mean()/x.std(),4) if len(x)>1 else None)
