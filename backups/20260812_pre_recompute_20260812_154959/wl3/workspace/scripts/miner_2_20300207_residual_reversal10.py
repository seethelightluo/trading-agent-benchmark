import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    d=None
    for fn in (get_stock_daily_data,get_index_daily_data):
        try: d=fn(s,days=4000)
        except Exception: d=None
        if d is not None and len(d)>100: break
    if d is not None and len(d)>100:
        z=d.copy(); z.date=pd.to_datetime(z.date); P[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); bench=r.mean(axis=1)
# market-neutral residual reversal: fade 10-session cumulative residual, risk scaled by idiosyncratic 60d vol
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(bench,axis=0),axis=0)
idvol=res.rolling(60,min_periods=40).std()
f=-(res.rolling(10,min_periods=8).sum())/(idvol*np.sqrt(10))
# signal is only available after completed date; evaluate dates f[t] vs returns t+1...
for h in [1,3,5,10]:
 vals=[]; dates=[]; ns=[]
 fr=f.shift(1); fw=r.shift(-h).rolling(h).sum().shift(-(h-1))
 for dt in f.index:
  a=fr.loc[dt]; b=fw.loc[dt]; q=pd.concat([a,b],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); dates.append(dt); ns.append(len(q))
 x=np.array(vals); print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/len(U))
# regimes by benchmark trailing 60 trend and vol terciles for daily
fr=f.shift(1); fw=r.shift(-1)
for name,mask in [('early',f.index<'2023-01-01'),('mid', (f.index>='2023-01-01')&(f.index<'2027-01-01')),('late',f.index>='2027-01-01'),('highvol',bench.rolling(60).std()>bench.rolling(60).std().quantile(.7)),('lowvol',bench.rolling(60).std()<bench.rolling(60).std().quantile(.3))]:
 vals=[]
 for dt in f.index[mask]:
  q=pd.concat([fr.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('REG',name,len(vals),np.nanmean(vals) if vals else np.nan)
print('instruments',len(P),'dates',len(p),'last',p.index[-1])
# artifact with signal and date/symbol for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20300207_residual_reversal10_signal.csv',index=False)
