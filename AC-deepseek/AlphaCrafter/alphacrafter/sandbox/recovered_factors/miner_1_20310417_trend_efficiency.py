import pandas as pd, numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; hi={}; lo={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[a]=d.close; hi[a]=d.high; lo[a]=d.low
p=pd.DataFrame(px); r=p.pct_change()
# Trend efficiency: signed cumulative return divided by cumulative absolute daily returns;
# rewards persistent directional travel while suppressing choppy momentum.
eff=r.rolling(30,min_periods=24).sum()/r.abs().rolling(30,min_periods=24).sum()
# use one-day lag because decisions see completed prior day
sig=eff.shift(1)
fwd=p.shift(-10)/p-1
all_ic=[]; ns=[]
for dt in p.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  all_ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
a=np.array(all_ic)
print('candidate=30d_trend_efficiency_lag1','dates',len(a),'meanN',round(np.mean(ns),2),'minN',min(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; nn=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
 q=np.array(vals); print('H',h,'dates',len(q),'N',round(np.mean(nn),2),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1),'hit',np.mean(q>0))
print('coverage',sig.notna().mean().mean(),'turn10',sig.rank(axis=1,pct=True).diff(10).abs().mean().mean())
for lo,hi in [(2020,2023),(2024,2027),(2028,2030),(2030,2031)]:
 vals=[]
 for dt in p.index:
  if lo<=dt.year<=hi:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REGIME',lo,hi,'dates',len(q),'IC',np.mean(q) if len(q) else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
# correlation audit against available admitted factor files cannot be inferred from definitions; explicitly report unavailable
print('max_abs_library_correlation=UNAVAILABLE')
