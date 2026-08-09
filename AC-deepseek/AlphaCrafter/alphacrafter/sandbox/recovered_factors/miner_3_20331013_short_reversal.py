import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym, macro=False):
 p=('../persistent/index_data/'+sym+'.csv') if macro else ('../persistent/stock_data/'+sym+'.csv')
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); return d.set_index('date').sort_index()
px=pd.DataFrame({a:load(a).close for a in assets})
vix=load('VIX',True).close.reindex(px.index).ffill()
r=np.log(px).diff(); vol=r.rolling(20).std()*np.sqrt(252)
# candidate: 5-day reversal scaled by trailing volatility, with a shock gate to avoid tiny moves
raw=-(px.pct_change(5)/(vol+1e-8))
# delayed one completed session
sig=raw.shift(1)
# only report dates with >=8 names
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1
 vals=[]; dates=[]; nms=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); nms.append(ok.sum())
 z=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0),len(z),np.mean(nms)))
# regimes for 10d
h=10; fwd=px.shift(-h)/px-1; vals=[]
for dt in sig.index:
 ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
 if ok.sum()>=8: vals.append((dt,spearmanr(sig.loc[dt][ok],fwd.loc[dt][ok]).statistic))
for a,b in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2033-12-31')]:
 z=np.array([v for d,v in vals if d>=pd.Timestamp(a) and d<=pd.Timestamp(b)]); print('REG',a,b,len(z),np.mean(z) if len(z) else np.nan,(np.mean(z)/np.std(z,ddof=1)) if len(z)>1 else np.nan)
print('coverage',sig.notna().sum().sum()/(len(sig)*15),'turnover',np.mean([np.nanmean((sig.iloc[i].rank(pct=True)-sig.iloc[i-10].rank(pct=True)).abs()) for i in range(10,len(sig))]))
print('visible',px.index.max())
