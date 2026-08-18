import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=6000)
            if x is not None and len(x)>300:return x
        except Exception: pass
raw={s:fetch(s) for s in U}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None},axis=1).sort_index()
r=np.log(p).diff()
# Candidate: medium-term trend per realized risk, with a short-term shock penalty.
vol=r.rolling(20,min_periods=10).std()*np.sqrt(252)
f=(r.rolling(30,min_periods=20).sum()/(vol+1e-12) - 0.35*r.rolling(3,min_periods=3).sum()/(r.rolling(10,min_periods=8).std()+1e-12)).rank(axis=1,pct=True).shift(1)
rows=[]
for h in [1,5,10,20,40]:
 out=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: out.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').loc['2026-07-16':'2034-06-22']
 if len(q):
  print('H',h,'dates',len(q),'inst_mean',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4),'turn',round((f.diff().abs().mean().mean()),6))
  for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2034-06-22')]:
   z=q.loc[a:b]; print(a,b,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6) if len(z)>1 else None)
print('coverage',f.notna().mean().mean(),'assets',len(p.columns),'dates',len(p))
# Save signal artifact for reproducibility
sig=f.copy(); sig.index=sig.index.strftime('%Y-%m-%d'); sig.to_csv('../persistent/miner_3_20340623_trend_shock_blend_signal.csv')
