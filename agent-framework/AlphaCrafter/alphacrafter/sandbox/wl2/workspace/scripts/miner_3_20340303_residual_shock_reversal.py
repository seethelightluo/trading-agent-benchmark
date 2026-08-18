import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Residual shock: remove contemporaneous equal-weight cross-asset market move,
# then reverse 10d residual move, scaled by idiosyncratic 40d risk and filtered for non-trending paths.
mkt=r.mean(axis=1); resid=r.sub(mkt,axis=0)
res10=resid.rolling(10).sum(); iv=resid.rolling(40).std()*np.sqrt(40)
trend=resid.rolling(60).sum()/(resid.abs().rolling(60).sum()+1e-12)
rawf=(-res10/(iv+1e-12)).clip(-4,4)
f=rawf.rank(axis=1,pct=True)*((1-trend.abs()).clip(0,1).rank(axis=1,pct=True)); f=f.shift(1)
rows=[]
for h in [10,20,40]:
 fr=p.shift(-h)/p-1; out=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: out.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-03-02']
 print('horizon',h,'dates',len(q),'assets',len(raw),'avgN',round(q.n.mean(),2),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-02'])]:
  print(lab,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1) if len(qq)>1 else np.nan)
 if h==20: z.reset_index().to_csv('scripts/miner_3_20340303_residual_shock_reversal_ic.csv',index=False)
f.to_csv('scripts/miner_3_20340303_residual_shock_reversal_signal.csv')
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
