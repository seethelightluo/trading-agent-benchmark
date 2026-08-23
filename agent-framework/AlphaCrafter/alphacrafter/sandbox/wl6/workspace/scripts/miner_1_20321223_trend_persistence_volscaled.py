import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-12-22']; r=p.pct_change()
# Trend persistence: medium trend, penalized by short-term disagreement and scaled by trailing risk.
# Positive when 20d trend and latest 5d move agree; disagreement suppresses the signal.
trend=p.pct_change(20); short=p.pct_change(5); vol=r.rolling(30).std()
sig=trend.div(vol.replace(0,np.nan)) * np.sign(trend*short)
for h in [5,10,20,40]:
 fwd=p.shift(-h).div(p)-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
 if h==10:
  print('coverage %.4f turnover %.4f'%(sig.notna().sum().sum()/sig.size,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
  print('annual',q.assign(year=q.index.year).groupby('year').ic.mean().round(5).to_dict())
