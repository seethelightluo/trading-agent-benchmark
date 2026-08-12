import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s, days=5000)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
raw=(r.rolling(5).sum()/v + r.rolling(10).sum()/v + r.rolling(20).sum()/v)/3
agree=((np.sign(r.rolling(5).sum())==np.sign(r.rolling(20).sum())).astype(float)*2-1); f=raw*agree
ics=[]; rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); rows.append((p.index[i],ics[-1],len(z)))
a=np.array(ics); print('dates',len(p),'instruments',len(D),'ic_dates',len(a),'avg_n',np.mean([x[2] for x in rows]),'coverage',len(a)/(len(p)-1)); print('IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
for name,lo,hi in [('2020-25','2020','2025-12-31'),('2026-29','2026','2029-12-31'),('2030-32','2030','2032-12-31'),('recent120',None,None)]:
 q=a[-120:] if name=='recent120' else np.array([x[1] for x in rows if str(x[0])[:10]>=lo and str(x[0])[:10]<=hi]); print(name,len(q),'%.6f %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)) if len(q)>1 else 'NA')
for h in [3,5,10]:
 yy=p.pct_change(h).shift(-h); aa=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],yy.iloc[i]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,len(aa),np.nanmean(aa))
out=f.copy(); out.index=out.index.astype(str); out.to_csv('scripts/miner_2_20320819_trend_agreement_signal.csv')
