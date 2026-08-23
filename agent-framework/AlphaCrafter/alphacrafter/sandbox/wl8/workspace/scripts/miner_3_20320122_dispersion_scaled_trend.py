import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items()}).sort_index().ffill()
# lag-safe: signal at t from close through t; forward starts t+1
r=px.pct_change(); mom10=px.pct_change(10); vol20=r.rolling(20).std(); disp=r.rolling(20).std().median(axis=1)
# cross-asset dispersion-conditioned: trend in low dispersion, short reversal component in high dispersion
# scale by volatility, with robust cross-sectional dispersion threshold
z=(disp-disp.rolling(120).median())/(disp.rolling(120).median()+1e-12)
trend=mom10/(vol20+1e-12)
rev=-r.rolling(3).sum()/(vol20+1e-12)
w=np.clip((z-0.05)/0.5,0,0.5)
f=trend.mul(1-w,axis=0)+rev.mul(w,axis=0)
fr=px.shift(-10)/px-1
rows=[]
for i,date in enumerate(px.index[:-10]):
    if i<150: continue
    a=f.loc[date]; b=fr.loc[date]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        rows.append((date,spearmanr(a[ok],b[ok]).statistic,ok.sum(),0))
x=pd.DataFrame(rows,columns=['date','ic','n','dummy']).set_index('date')
# daily and non-overlap, both explicitly reported
for name,y in [('daily',x),('nonoverlap',x.iloc[::10])]:
    ic=y.ic.dropna(); print(name,'dates',len(ic),'avg_n',y.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for h in [1,5,10,20]:
 frh=px.shift(-h)/px-1; vals=[]
 for date in px.index[:-h]:
  a=f.loc[date]; b=frh.loc[date]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic)
 print('horizon',h,'IC',np.nanmean(vals),'n',len(vals))
print('period',x.index.min().date(),x.index.max().date(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('recent365',x.tail(365).ic.mean(),x.tail(365).ic.mean()/x.tail(365).ic.std(ddof=1))
print('year',x.groupby(x.index.year).ic.agg(['mean','std']).assign(icir=lambda q:q['mean']/q['std']))
