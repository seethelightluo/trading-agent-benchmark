import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Candidate: calm-regime risk-normalized 20d momentum; VIX percentile is lagged/rolling and fully observable.
vol=r.rolling(20).std()*np.sqrt(252)
mom=p.pct_change(20)
vixpct=vix.rolling(120,min_periods=60).rank(pct=True)
f=mom.div(vol).mul(1-vixpct,axis=0)
# forward ten trading day simple return
fr=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 x=f.loc[d]; y=fr.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# Require dates with 15 names, report both 10d IC and daily ICIR convention used by gate.
print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15)
print('IC',df.ic.mean(),'ICIR',df.ic.mean()/df.ic.std(ddof=1),'hit',(df.ic>0).mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=df.loc[a:b]
 print(a+'-'+b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
# cross-sectional rank turnover
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(10)).abs().mean(axis=1).dropna()
print('turnover',turn.mean(),'latest',df.index[-1].date())
# decay 5/10/20
for h in [5,10,20]:
 rr=p.shift(-h).div(p)-1; vals=[]
 for d in f.index:
  z=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.nanmean(vals),'n',len(vals))
# artifact for audit
out=pd.DataFrame({s:f[s] for s in U}); out.to_csv('scripts/miner_2_20331111_vix_calm_momentum_signal.csv',index_label='date')
