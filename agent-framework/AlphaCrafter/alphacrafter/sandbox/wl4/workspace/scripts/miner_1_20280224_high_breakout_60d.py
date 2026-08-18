import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv'); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change()
# distance from rolling 60-day high, expressed as high proximity (higher is bullish)
factor=prices/prices.rolling(60,min_periods=50).max()-1
fwd=prices.shift(-10)/prices-1
ics=[]; turnovers=[]; ninst=[]
for dt in factor.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); ninst.append(ok.sum())
  if len(ics)>1:
   prev=factor.iloc[factor.index.get_loc(dt)-1]; z=prev.notna()&x.notna()
   turnovers.append(np.mean(np.abs(x[z].rank(pct=True)-prev[z].rank(pct=True))))
a=np.array(ics); mean=np.nanmean(a); sd=np.nanstd(a,ddof=1)
print({'factor':'60d_high_proximity','dates':len(a),'avg_instruments':round(np.mean(ninst),2),'coverage':round(np.mean(ninst)/15,4),'ic':round(mean,6),'icir':round(mean/sd*np.sqrt(252),6),'hit':round(np.mean(a>0),4),'turnover':round(np.mean(turnovers),4),'period':f'{factor.index.min().date()} to {factor.index.max().date()}'})
for h in [1,5,10,20]:
 yy=prices.shift(-h)/prices-1; z=[]
 for dt in factor.index:
  ok=factor.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(factor.loc[dt][ok],yy.loc[dt][ok]).statistic)
 z=np.array(z); print('decay',h,round(np.nanmean(z),6),round(np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252),6))
