import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: D[s]=get_index_daily_data(s,days=4000)
    except Exception: D[s]=get_stock_daily_data(s,days=4000)
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill()
r=px.pct_change(); vol=r.rolling(40).std(); down=r.where(r<0,0).rolling(40).std()
# Prefer assets with less downside variability relative to total risk; lag one session.
f=(-(down/(vol+1e-12))).shift(1)
rows=[]
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=q.ic
 print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f recent365=%.6f/%.6f'%(h,len(q),q.n.mean(),q.n.mean()/15,ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),ic.tail(365).mean(),ic.tail(365).mean()/ic.tail(365).std(ddof=1)))
if len(f)>1:
 turn=[f.iloc[i].rank(pct=True).sub(f.iloc[i-1].rank(pct=True)).abs().mean() for i in range(1,len(f))]
 print('turnover=%.6f'%np.nanmean(turn))
