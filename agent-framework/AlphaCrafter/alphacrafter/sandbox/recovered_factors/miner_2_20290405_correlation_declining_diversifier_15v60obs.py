import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# One idea: correlation-declining diversifier signal.  An asset is favored when its
# recent average correlation to the rest of the cross-asset universe has fallen
# versus its longer-run correlation; this is a portfolio-structure feature, not a price reversal.
END='2029-04-04'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for a in assets:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 cl[a]=d.close
close=pd.DataFrame(cl).sort_index(); ret=close.pct_change()
# For each date and asset, compute mean correlation with other assets, then short-minus-long.
# Negative values mean correlation has declined, providing a potentially useful diversifier.
sig=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for k in range(59,len(ret)):
 w=ret.iloc[k-59:k+1]
 long=w.corr(min_periods=40); short=w.iloc[-15:].corr(min_periods=10)
 for a in assets:
  peers=[x for x in assets if x!=a]
  sig.iloc[k,sig.columns.get_loc(a)]=long.loc[a,peers].mean()-short.loc[a,peers].mean()
def stats(s,h):
 vals=[]; ns=[]; fwd=close.shift(-h)/close-1
 for t in s.index:
  x=s.loc[t]; y=fwd.loc[t]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 z=np.array(vals); return {'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)) if len(z)>1 and z.std(ddof=1)>0 else None,'hit':float((z>0).mean()),'dates':len(z),'mean_instruments':float(np.mean(ns))}
print('idea correlation-declining diversifier, endpoint',END)
print('coverage',float(sig.notna().mean().mean()),'signal_dates',len(sig))
for h in [1,5,10,20]: print('horizon',h,json.dumps(stats(sig,h)))
for name,mask in [('2026_2028',sig.index<='2028-12-31'),('2029_ytd',sig.index>='2029-01-01')]: print('regime',name,json.dumps(stats(sig.loc[mask],5)))
pairs=[]
for i in range(1,len(sig)):
 a,b=sig.iloc[i-1],sig.iloc[i]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:pairs.append(spearmanr(a[ok],b[ok]).statistic)
print('rank_stability',float(np.nanmean(pairs)),'turnover',float(1-np.nanmean(pairs)),'pairs',len(pairs))
out='scripts/miner_2_20290405_correlation_declining_diversifier_15v60obs_signal.pkl';sig.to_pickle(out)
effective=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f))
  if j.get('validation',{}).get('status')=='EFFECTIVE':effective.append(j['factor_id'])
 except:pass
maxrho=0.; evidence=[]; missing=[]
for fid in effective:
 m=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not m: missing.append(fid);continue
 try:
  x=pd.read_pickle(m[-1]);x.index=pd.to_datetime(x.index)
  rs=[]
  for t in sig.index.intersection(x.index):
   a=sig.loc[t,sig.columns.intersection(x.columns)];b=x.loc[t,a.index];ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:rs.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if rs: evidence.append((fid,float(max(rs)),len(rs)));maxrho=max(maxrho,max(rs))
  else:missing.append(fid)
 except Exception:missing.append(fid)
print('LIBRARY_CORRELATION',json.dumps({'audited':len(evidence),'effective':len(effective),'max_abs_library_correlation':maxrho,'evidence':evidence,'missing':missing}))
