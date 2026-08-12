import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
D={}
for s in U:
 d=get_stock_daily_data(symbol=s,days=4000)
 if d is not None:
  d=d.sort_values('date'); D[s]=d.set_index('date')['close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in D.values()]))
P=pd.DataFrame({s:D[s].reindex(common) for s in U},index=common).ffill()
R=P.pct_change(); fwd=P.shift(-1)/P-1
# common return proxy and breadth computed at t, signal lagged one session via shift
m20=P.pct_change(20); breadth=(m20>0).mean(axis=1)
# stress = weak breadth; reversal only activated on weak breadth, residualize 5d return cross section
r5=P.pct_change(5); csmed=r5.median(axis=1)
res=r5.sub(csmed,axis=0)
vol=R.rolling(20).std().replace(0,np.nan)
raw=-res/vol
# smooth stress gate, activated weak breadth; signal artifact at date t uses t data then evaluated next day
sig=raw.where(breadth<.40, np.nan).shift(1)
rows=[]; ics=[]; active=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); active.append(len(z)); rows.append([dt,*x.values])
ics=np.array(ics,float)
print('dates',len(ics),'avgN',np.mean(active),'active fraction',np.isfinite(sig).sum().sum()/sig.size)
print('1d IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0)))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); print('%dd n=%d IC %.6f ICIR %.6f avgN %.2f'%(h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(ns)))
# turnover among active signal ranks
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('turnover',turn,'coverage',np.isfinite(sig).mean().mean())
out=pd.DataFrame(rows,columns=['date']+U); out.to_csv('scripts/miner_3_20300725_stress_residual_reversal_signal.csv',index=False)
