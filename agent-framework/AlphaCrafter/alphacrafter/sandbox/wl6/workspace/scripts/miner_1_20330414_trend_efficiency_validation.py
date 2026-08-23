import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2033-04-13')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).query('date<=@end').sort_values('date'); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Trend efficiency: net 20d move divided by realized path, rewarding persistent directional movement.
net=R.rolling(20,min_periods=20).sum(); path=R.abs().rolling(20,min_periods=20).sum()+1e-8
F=net/path
ics=[]; ns=[]; dates=[]; turns=[]; prev=None
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; x=F.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z):
   ics.append(z);ns.append(int(ok.sum()));dates.append(dt)
   r=x.rank(pct=True)
   if prev is not None: turns.append(float((r-prev).abs().mean()))
   prev=r
A=np.array(ics); print(json.dumps({'factor':'trend_efficiency_20d','dates':len(A),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_n':np.mean(ns),'coverage':np.mean(ns)/15,'ic':np.mean(A),'icir':np.mean(A)/(np.std(A,ddof=1)+1e-12)*np.sqrt(252/10),'hit':np.mean(A>0),'turnover':np.mean(turns)},indent=2))
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; aa=[]
 for dt in F.index:
  ok=F.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(F.loc[dt][ok],y.loc[dt][ok]).statistic
   if np.isfinite(z):aa.append(z)
 print('decay',h,np.mean(aa),len(aa))
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2033-04-13')]:
 q=[v for d,v in zip(dates,ics) if pd.Timestamp(a)<=d<=pd.Timestamp(b)]; print('regime',a,len(q),np.mean(q) if q else None)
F.reset_index().rename(columns={'date':'timestamp'}).to_csv('scripts/miner_1_20330414_trend_efficiency_signal.csv',index=False)
