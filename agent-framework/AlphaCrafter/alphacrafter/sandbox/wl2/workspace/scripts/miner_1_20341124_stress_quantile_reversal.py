import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
TODAY=pd.Timestamp('2034-11-24'); base='../persistent/stock_data'; macro='../persistent/index_data'
def load(p):
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close.astype(float)
px=pd.concat({a:load(f'{base}/{a}.csv') for a in assets},axis=1).sort_index().loc[:TODAY]
vix=load(f'{macro}/VIX.csv').reindex(px.index).ffill(); dxy=load(f'{macro}/DXY.csv').reindex(px.index).ffill()
r=np.log(px).diff(); rev=-(np.log(px.shift(1))-np.log(px.shift(16)))
vol=r.shift(1).rolling(15,min_periods=10).std()*np.sqrt(252); raw=rev/vol
# A less sparse stress state: VIX above trailing 75th percentile and DXY positive over 10 days.
vixq=vix.shift(1).rolling(252,min_periods=126).quantile(.75)
active=(vix.shift(1)>vixq)&(dxy.shift(1).pct_change(10)>0)
sig=raw.mask(~active,axis=0); fwd=np.log(px.shift(-10)/px); rows=[]
for dt in px.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
for label,q in [('all',z),('recent',z[z.index>=pd.Timestamp('2029-01-01')])]:
 print(label,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('coverage',round(sig.notna().sum().sum()/sig.size,4),'active_dates',int((sig.notna().sum(axis=1)>=8).sum()),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'period',z.index.min().date(),z.index.max().date())
for h in [5,20,40]:
 yy=np.log(px.shift(-h)/px); rr=[]
 for dt in px.index:
  ok=sig.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8: rr.append(spearmanr(sig.loc[dt][ok],yy.loc[dt][ok]).statistic)
 rr=pd.Series(rr).dropna(); print('decay',h,round(rr.mean(),6),round(rr.mean()/rr.std(ddof=1),6),len(rr))
sig.index.name='date'; sig.to_csv('../persistent/miner_1_20341124_stress_quantile_reversal_signal.csv')
