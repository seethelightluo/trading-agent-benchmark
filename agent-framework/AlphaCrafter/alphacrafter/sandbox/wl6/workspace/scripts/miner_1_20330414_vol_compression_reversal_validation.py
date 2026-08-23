import os,json,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2033-04-13')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).query('date<=@end').sort_values('date'); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); v5=R.rolling(5).std(); v20=R.rolling(20).std()+1e-8
# Buy recent losers only when short volatility is compressed versus its medium baseline.
F=-(R.rolling(10).sum())*(v20/v5).clip(0,5)
fr=P.shift(-10)/P-1; ic=[]; ns=[]; dates=[]; turns=[]; prev=None
for dt in F.index:
 x=F.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z):
   ic.append(z);ns.append(ok.sum());dates.append(dt); rr=x.rank(pct=True)
   if prev is not None:turns.append((rr-prev).abs().mean())
   prev=rr
A=np.array(ic); print(json.dumps({'factor':'vol_compression_reversal_10d','dates':len(A),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_n':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'ic':float(np.mean(A)),'icir':float(np.mean(A)/(np.std(A,ddof=1)+1e-12)*np.sqrt(252/10)),'hit':float(np.mean(A>0)),'turnover':float(np.mean(turns))},indent=2))
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; a=[]
 for dt in F.index:
  ok=F.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(F.loc[dt][ok],y.loc[dt][ok]).statistic
   if np.isfinite(z):a.append(z)
 print('decay',h,float(np.mean(a)),len(a))
for a,b in [('2026','2028-12-31'),('2029','2030-12-31'),('2031','2033-04-13')]:
 q=[v for d,v in zip(dates,ic) if pd.Timestamp(a)<=d<=pd.Timestamp(b)];print('regime',a,len(q),float(np.mean(q)) if q else None)
F.reset_index().rename(columns={'date':'timestamp'}).to_csv('scripts/miner_1_20330414_vol_compression_reversal_signal.csv',index=False)
