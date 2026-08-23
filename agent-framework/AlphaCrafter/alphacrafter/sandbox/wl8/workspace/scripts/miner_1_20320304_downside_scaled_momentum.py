import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-03-04')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.close[d.index<=cut]
P=pd.DataFrame(px).sort_index()
r=P.pct_change()
# interpretable: medium momentum penalized by downside volatility; bounded via rank cross-section
raw=P.pct_change(20)/(r.where(r<0,0).pow(2).rolling(20).mean().pow(.5)+1e-8)
# standardize each date cross-section; forward 10 trading-day return
fwd=P.shift(-10)/P-1
ics=[]; cover=[]; turns=[]
prev=None
for dt in raw.index:
 x=raw.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); cover.append(ok.sum()/15)
  ranks=x.rank(pct=True)
  if prev is not None: turns.append((ranks-prev).abs().mean())
  prev=ranks
ics=np.array(ics)
print('factor=downside_scaled_momentum_20d dates',len(ics),'instruments=15')
print('IC_mean',ics.mean(),'ICIR',ics.mean()/(ics.std(ddof=1)+1e-12),'hit',np.mean(ics>0),'coverage',np.mean(cover),'turnover',np.mean(turns))
for name,mask in [('2021-2023',(np.arange(len(ics))<len(ics)//3)),('2024-2026',(np.arange(len(ics))>=len(ics)//3)&(np.arange(len(ics))<2*len(ics)//3)),('2027-2032',(np.arange(len(ics))>=2*len(ics)//3))]:
 z=ics[mask]; print(name,len(z),z.mean(),z.mean()/(z.std(ddof=1)+1e-12),np.mean(z>0))
# decay horizons
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; a=[]
 for dt in raw.index:
  ok=raw.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:a.append(spearmanr(raw.loc[dt][ok],yy.loc[dt][ok]).statistic)
 a=np.array(a); print('decay',h,a.mean(),a.mean()/(a.std(ddof=1)+1e-12),len(a))
