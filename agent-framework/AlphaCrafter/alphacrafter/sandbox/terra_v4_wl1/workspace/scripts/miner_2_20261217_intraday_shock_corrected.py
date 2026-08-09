import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-16'); base='../persistent/stock_data'
F={}; Y={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 close=d.close.astype(float); op=d.open.astype(float); hi=d.high.astype(float); lo=d.low.astype(float)
 prev=close.shift(1)
 tr=pd.concat([(hi-lo),(hi-prev).abs(),(lo-prev).abs()],axis=1).max(axis=1)
 atr=tr.shift(1).rolling(20,min_periods=15).mean()
 # prior completed intraday move, normalized by prior ATR; fade shock
 F[s]=(-(close-op)/atr).rename(s)
 Y[s]=close.shift(-1).div(close)-1
f=pd.concat(F,axis=1).sort_index(); y=pd.concat(Y,axis=1).reindex(f.index)
for h in [1,5,10]:
 if h==1: yy=y
 else: yy=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut].close.astype(float).shift(-h).div(pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut].close.astype(float))-1 for s in U},axis=1).reindex(f.index)
 vals=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append((dt,spearmanr(q.f,q.y).statistic,len(q)))
 a=pd.DataFrame(vals,columns=['date','ic','n']); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in a.groupby(a.date.dt.year): print('YR',yr,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'cut',cut)
# audit artifact
f.to_csv('scripts/miner_2_20261217_intraday_shock_signal.csv')
