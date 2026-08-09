import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); px[a]=d['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Acceleration: recent 20d excess return relative to its prior 40d pace, volatility normalized; lag one day.
r20=p.pct_change(20); r60=p.pct_change(60)
acc=r20-r60/3
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(acc/vol).shift(1)
print('rows=%d assets=%d date=%s..%s'%(len(p),len(assets),p.index.min().date(),p.index.max().date()))
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; dates=[]; nvalid=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic); dates.append(dt); nvalid.append(ok.sum())
 s=pd.Series(vals,index=dates); print('h=%d dates=%d mean_valid=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(nvalid),s.mean(),s.mean()/s.std(ddof=1), (s>0).mean()))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-09-19')]:
  q=s.loc[lo:hi]; print(' regime=%s n=%d IC=%.6f ICIR=%.6f'%(lo,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan))
rank=f.rank(axis=1,pct=True); print('coverage=%.4f turnover10=%.4f mean_valid=%.2f'%(f.notna().sum().sum()/f.size,(rank-rank.shift(10)).abs().mean(axis=1).mean(),f.notna().sum(axis=1).mean()))
