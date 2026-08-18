import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-04-18'); base='../persistent/stock_data'; px={}
for a in assets:
 p=f'{base}/{a}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date'); px[a]=d.loc[d.index<=cut,'close']
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change(); bench=rets.mean(axis=1)
rows=[]
for date in prices.index:
 if len(prices.loc[:date])<80: continue
 r20=prices.loc[date]/prices.shift(20).loc[date]-1; vol=rets.rolling(20).std().loc[date]*np.sqrt(20)
 br=(1+bench.loc[:date].tail(20).fillna(0)).prod()-1
 for a in assets:
  if a not in prices: continue
  rr=rets[a].loc[:date].tail(60); bb=bench.loc[rr.index]
  beta=rr.cov(bb)/bb.var() if len(rr.dropna())>=40 and bb.var()>1e-12 else np.nan
  residual=(r20[a]-beta*br) if pd.notna(beta) else np.nan
  f=-residual/vol[a] if pd.notna(vol[a]) and vol[a]>0 else np.nan
  rows.append((date,a,f))
s=pd.DataFrame(rows,columns=['date','asset','factor']).pivot(index='date',columns='asset',values='factor')
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; ics=[]; ns=[]
 for d in s.index:
  z=pd.concat([s.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(ics); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),5),'hit',round(np.mean(a>0),3),'coverage',round(np.mean(ns)/15,4))
# rank turnover proxy, 20d signal
r=s.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).mean(); print('turnover_proxy',round(float(turn),5)); print('period',s.index.min().date(),s.index.max().date(),'assets',len(px))
