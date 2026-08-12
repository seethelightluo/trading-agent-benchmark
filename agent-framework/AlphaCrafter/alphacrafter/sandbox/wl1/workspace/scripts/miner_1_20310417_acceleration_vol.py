import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); prices[a]=d.set_index('date')['close']
px=pd.DataFrame(prices).sort_index().ffill(); rets=px.pct_change()
# interpretable medium-term acceleration, volatility normalized and cross-sectional demeaned
r20=px.pct_change(20); r60=px.pct_change(60); vol=rets.rolling(40).std()*np.sqrt(40)
f=(r20-0.5*r60)/vol
# lag one completed day
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 fr=px.pct_change(h).shift(-h)
 ics=[]; dates=[]; cov=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); cov.append(ok.mean())
 s=pd.Series(ics,index=pd.to_datetime(dates)); ic=s.mean(); sd=s.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 print('H',h,'dates',len(s),'avgN',round(np.mean([int((f.loc[d].notna()&fr.loc[d].notna()).sum()) for d in dates]),2),'IC',round(ic,6),'ICIR',round(icir,6),'hit',round((s>0).mean(),4),'coverage',round(np.mean(cov),4))
 for label,mask in [('2020-25',s.index<'2026-01-01'),('2026+',s.index>='2026-01-01'),('2029+',s.index>='2029-01-01'),('2030+',s.index>='2030-01-01')]:
  z=s[mask]; print(label,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(252),6) if len(z)>2 else np.nan)
# turnover ranks daily
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).mean()
print('turnover',turn)
f.to_csv('scripts/miner_1_20310417_accel_vol_signal.csv')
