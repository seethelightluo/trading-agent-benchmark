import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close.replace(0,np.nan)
P=pd.DataFrame(px); R=P.pct_change();
# defensive carry: recent return penalized by downside volatility, with cross-sectional demeaning
ret=P/P.shift(20)-1; down=R.where(R<0).rolling(40,min_periods=20).std(); f=ret.div(down.replace(0,np.nan)); f=f.sub(f.median(axis=1),axis=0)
ics=[]; ns=[]
for dt in f.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([f.loc[dt],y],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
ics=pd.Series(ics).dropna(); print('factor=defensive_carry_20d dates=%d avg_n=%.2f coverage=%.4f'%(len(ics),np.mean(ns),len(ics)/len(f)))
print('IC=%.6f ICIR=%.6f hit=%.4f std=%.6f'%(ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean(),ics.std(ddof=1)))
print('date_range',P.index.min(),P.index.max())
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay_%dd=%.6f n=%d'%(h,np.nanmean(vals),len(vals)))
