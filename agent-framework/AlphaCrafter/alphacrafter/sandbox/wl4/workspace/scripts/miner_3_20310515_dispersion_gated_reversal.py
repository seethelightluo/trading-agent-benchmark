import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=get_index_daily_data(s,days=4000)
 except Exception:D[s]=get_stock_daily_data(s,days=4000)
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=px.pct_change()
# reversal is activated only when cross-sectional 20d return dispersion is elevated
base=-(r.rolling(5).sum()-r.rolling(5).sum().median(axis=1).values[:,None]); disp=r.rolling(20).std().std(axis=1)
gate=(disp>disp.rolling(252,min_periods=60).median()).astype(float); f=(base*gate.values[:,None]).shift(1)
fw=px.shift(-10)/px-1; rows=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:rows.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=res.ic; turn=[]
for i in range(1,len(f)): turn.append(f.iloc[i].rank(pct=True).sub(f.iloc[i-1].rank(pct=True)).abs().mean())
print('factor=dispersion_gated_residual_reversal dates=%d avg_n=%.2f coverage=%.4f'%(len(res),res.n.mean(),res.n.mean()/15));print('IC=%.6f ICIR=%.6f hit=%.4f turnover=%.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),np.nanmean(turn)))
for w in [365,730,1095]:
 q=ic.tail(w);print('recent%d IC=%.6f ICIR=%.6f hit=%.4f dates=%d'%(w,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1;a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay_h%d IC=%.6f dates=%d'%(h,np.nanmean(a),len(a)))
