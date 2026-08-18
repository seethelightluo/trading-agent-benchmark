import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-07-22')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; D[s]=x[x.index<=CUT]
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
res10=resid.rolling(10,min_periods=8).sum(); down=r.clip(upper=0).pow(2).rolling(30,min_periods=20).mean().pow(.5)*np.sqrt(252)
sig=-(res10/down.replace(0,np.nan)).shift(1)
def calc(mask,h=20):
 f=p.shift(-h)/p-1; vals=[]; turns=[]; ns=[]; prev=None
 for d in sig.index[mask]:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); rk=q.iloc[:,0].rank(pct=True)
   if prev is not None: turns.append(np.mean(abs(rk-prev)))
   prev=rk
 a=np.array(vals); return len(a),np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),np.mean(ns)/15,np.nanmean(turns)
for h in [5,10,20]:
 # reconstruct forward each horizon within cutoff naturally
 f=p.shift(-h)/p-1; vals=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));rk=q.iloc[:,0].rank(pct=True)
   if prev is not None:turns.append(np.mean(abs(rk-prev)))
   prev=rk
 a=np.array(vals); print('H',h,'N IC ICIR hit coverage turnover',len(a),round(np.nanmean(a),6),round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),round(np.mean(a>0),4),round(np.mean(ns)/15,4),round(np.nanmean(turns),5))
for label,mask in [('2020-27',sig.index<'2028-01-01'),('2028-30',(sig.index>='2028-01-01')&(sig.index<'2031-01-01')),('2031-32',sig.index>='2031-01-01'),('recent365',sig.index>=CUT-pd.Timedelta(days=365))]: print(label,calc(mask))
print('last',p.index.max().date())
