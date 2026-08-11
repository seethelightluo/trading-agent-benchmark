import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Gap reversal: fade overnight open-to-prior-close move, using only completed session OHLC.
F=pd.DataFrame({s:-(D[s].open/D[s].close.shift(1)-1) for s in U}).sort_index()
outs={}
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:
   q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));dates.append(dt)
 q=np.array(q); outs[h]=q
 print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for yr in range(2020,2027):
 q=[]
 for dt in F.loc[str(yr)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':pd.DataFrame({s:D[s].close.shift(-1)/D[s].close-1 for s in U}).loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(q),'IC',round(np.mean(q),6) if q else None)
# rank turnover/coverage
print('coverage',round(F.notna().sum().sum()/F.size,4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# recent rolling blocks
q=outs[1]
for k in [252,504,756]:
 x=q[-k:];print('recent',k,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'dates',len(x))
