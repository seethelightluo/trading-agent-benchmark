import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2035-02-28'
P={}; H={}; L={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 P[a]=pd.to_numeric(d.close,errors='coerce'); H[a]=pd.to_numeric(d.high,errors='coerce'); L[a]=pd.to_numeric(d.low,errors='coerce')
p=pd.DataFrame(P); hi=pd.DataFrame(H); lo=pd.DataFrame(L)
# Close-location impulse: range-normalized signed candle, smoothed over 3 days.
rng=(hi-lo).replace(0,np.nan); loc=((p-lo)/rng-0.5).clip(-.5,.5)
atr=rng.rolling(20,min_periods=10).median()
# continuation signal: strong closes near high/low with range expansion, risk-normalized
F=(loc*(rng/atr).clip(0,3)).rolling(3,min_periods=3).mean()
print('idea=range_normalized_close_location_continuation')
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z)); ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'N',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for name,m in [('20-24',(s.index.year<=2024)),('25-29',(s.index.year>=2025)&(s.index.year<=2029)),('30-32',(s.index.year>=2030)&(s.index.year<=2032)),('33-35',s.index.year>=2033)]:
  q=s[m]; print(' ',name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
rank=F.rank(pct=True); print('coverage',round(F.notna().mean().mean(),4),'meanN',round(F.notna().sum(axis=1).mean(),2),'turn',round(rank.diff().abs().mean(axis=1).mean(),4),'cells',int(F.notna().sum().sum()),'cutoff',cut)
# decay from same common dates is represented per horizon above
