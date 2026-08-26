import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-07-25')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# Pullback-in-trend: short-term relative weakness is rewarded only when the
# asset retains a positive medium-term trend; all inputs are lagged one day.
ret5=p.pct_change(5); ret60=p.pct_change(60); med5=ret5.median(axis=1); med60=ret60.median(axis=1)
down=r.clip(upper=0).abs().rolling(30,min_periods=15).mean()
weak=(ret5.sub(med5,axis=0)).clip(upper=0)
trend=(ret60.sub(med60,axis=0)).clip(lower=0)
f=(-weak.div(down+1e-8)* (1+trend.clip(upper=.5))).shift(1)
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((dt,q,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); ic=a.ic.mean(); ir=ic/(a.ic.std(ddof=1)+1e-12)
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
  if len(sl): print('regime',h,name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
rank=f.rank(axis=1,pct=True); print('turnover_proxy',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_1_20300725_pullback_trend_signal.csv')
