import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[a]=d['close'].replace(0,np.nan)
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# sign-consistency / hit-rate trend: lagged fraction of positive daily returns, centered at 0
# This is intentionally not a return magnitude factor; it tests persistence of direction.
sig=(ret.rolling(20,min_periods=15).mean()).shift(1)
# normalize cross-sectionally to remove common breadth level
sig=sig.sub(sig.mean(axis=1),axis=0)
print('cutoff',close.index.max().date(),'dates',len(close),'assets',len(assets))

def ic(h):
 f=sig; fr=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 x=np.asarray(vals); return len(x),np.mean(ns),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12),np.mean(x>0), dates, x
for h in [1,5,10,20]:
 n,m,mu,ir,hit,dates,x=ic(h); print('H%d dates=%d meanN=%.2f IC=%+.6f ICIR=%+.6f hit=%.3f'%(h,n,m,mu,ir,hit))
 if n:
  for lo,hi in [('2020','2024'),('2025','2027'),('2028','2030'),('2030-04','2030-07')]:
   q=[v for d,v in zip(dates,x) if str(d)[:len(lo)]>=lo and str(d)[:len(hi)]<=hi] if '-' not in lo else [v for d,v in zip(dates,x) if d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi+'-31')]
   if q: print(' ',lo+'..'+hi,'n',len(q),'IC',round(float(np.mean(q)),6),'IR',round(float(np.mean(q)/(np.std(q,ddof=1)+1e-12)),4))
# turnover proxy, coverage
r1=sig.rank(axis=1,pct=True); turn=(r1-r1.shift(10)).abs().mean().mean(); cov=sig.notna().sum().sum()/sig.size
print('coverage=%.4f turnover10=%.4f'%(cov,turn))
# Comparator redundancy (not admission library audit)
for name,x in {'ret20':ret.rolling(20).mean().shift(1),'ret5':ret.rolling(5).mean().shift(1),'vol20':ret.rolling(20).std().shift(1)}.items():
 z=pd.concat([sig.stack(),x.stack()],axis=1).dropna(); print('pooled_spearman',name,round(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic),6),'cells',len(z))
print('NOTE: exact pooled audit against all admitted factor signals not reconstructed; candidate is not eligible for persistence.')
