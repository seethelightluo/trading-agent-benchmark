import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: D[s]=get_index_daily_data(s,days=4000)
 except Exception: D[s]=get_stock_daily_data(s,days=4000)
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=px.pct_change()
# lagged 20d return rewarded, normalized only by downside volatility; mild consistency multiplier
ret=px.pct_change(20); down=r.where(r<0,0).pow(2).rolling(20).mean().pow(.5); consistency=(r>0).rolling(20).mean()
f=(ret/down.replace(0,np.nan)*(.75+.5*consistency)).shift(1)
rows=[]
for i in range(len(px)-20):
 fw=px.shift(-10)/px-1; z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).dropna()
 if len(z)>=8: rows.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=res.ic
turn=[]
for i in range(1,len(px)):
 a=f.iloc[i].rank(pct=True); b=f.iloc[i-1].rank(pct=True); turn.append((a-b).abs().mean())
print('factor=downside_adjusted_momentum_20d dates=%d avg_n=%.2f coverage=%.4f'%(len(res),res.n.mean(),res.n.mean()/15)); print('H10 IC=%.6f ICIR=%.6f hit=%.4f turnover=%.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),np.nanmean(turn)))
for w in [365,730,1095]:
 q=ic.tail(w); print('recent%d IC=%.6f ICIR=%.6f hit=%.4f dates=%d'%(w,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1;a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay_h%d IC=%.6f dates=%d'%(h,np.nanmean(a),len(a)))
