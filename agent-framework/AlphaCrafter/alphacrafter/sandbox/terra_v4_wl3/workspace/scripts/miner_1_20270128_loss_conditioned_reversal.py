import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]; sig=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d['r3']=d.close.pct_change(3)
 # amplify reversal for assets with negative recent performance; retain signed signal for full ranking
 d['factor']=-d.r3*(1+1*(d.r3<0))
 d['fwd1']=d.close.shift(-1)/d.close-1
 d['fwd5']=d.close.shift(-5)/d.close-1
 d['fwd10']=d.close.shift(-10)/d.close-1
 for _,x in d.iterrows(): rows.append((x.date,s,x.factor,x.fwd1,x.fwd5,x.fwd10))
 for _,x in d[['date','factor']].dropna().iterrows(): sig.append((x.date,s,x.factor))
a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd1','fwd5','fwd10'])
def calc(col):
 z=a.dropna(subset=['factor',col]); vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g[col].nunique()>1: vals.append((dt,spearmanr(g.factor,g[col]).statistic,len(g)))
 q=pd.DataFrame(vals,columns=['date','ic','n']); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1); return len(q),q.n.mean(),ic,ir,(q.ic>0).mean()
print('candidate loss-conditioned 3d reversal; dates/instruments')
for c in ['fwd1','fwd5','fwd10']: print(c,calc(c))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 z=a[(a.date.dt.year>=int(lo))&(a.date.dt.year<=int(hi))]
 old=a; a=z
 print('regime',lo+'-'+hi,calc('fwd1'))
 a=old
# turnover as rank signal changes on common dates
wide=pd.DataFrame(sig,columns=['date','symbol','factor']).pivot(index='date',columns='symbol',values='factor')
ranks=wide.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print('coverage',a.factor.notna().mean(),'turnover',turnover,'avg names',a.dropna(subset=['factor']).groupby('date').size().mean())
# independent artifact for deterministic audit
wide.to_csv('scripts/miner_1_20270128_loss_conditioned_reversal_signal.csv')
