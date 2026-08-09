import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
 # gap reversal: yesterday close to today's open, known at today's close; predict next close return
 x['gap']=-(x['open']/x['close'].shift(1)-1)
 x['fwd']=x['close'].shift(-1)/x['close']-1
 D[s]=x[['gap','fwd']]
rows=[]
for s,x in D.items():
 z=x.copy(); z['symbol']=s; rows.append(z.reset_index())
a=pd.concat(rows).dropna(subset=['gap','fwd'])
ics=[]; ranks=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  
  z=spearmanr(g.gap,g.fwd).statistic
  if np.isfinite(z): ics.append(z)
  ranks.append(g.assign(r=g.gap.rank()).set_index('symbol')['r'])
ics=np.array(ics); print('dates',len(ics),'avg_names',len(a)/a.date.nunique(),'coverage',len(a)/(sum(len(x) for x in D.values())))
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0),'std',np.std(ics,ddof=1))
# horizons via forward close returns
for h in [5,10]:
 rr=[]
 for s,x in D.items():
  q=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
  q['gap']=-(q.open/q.close.shift(1)-1); q['fwd']=q.close.shift(-h)/q.close-1; rr.append(q[['gap','fwd']].reset_index())
 b=pd.concat(rr).dropna()
 ii=[spearmanr(g.gap,g.fwd).statistic for _,g in b.groupby('date') if len(g)>=8 and g.gap.nunique()>1 and g.fwd.nunique()>1]
 ii=[z for z in ii if np.isfinite(z)]
 print(h,'IC',np.mean(ii),'ICIR',np.mean(ii)/np.std(ii,ddof=1),'dates',len(ii))
# rank turnover
print('turnover',np.mean([np.mean((r-ranks[i-1]).abs())/len(r) for i,r in enumerate(ranks) if i]))
