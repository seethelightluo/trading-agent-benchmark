"""One candidate: continuous volatility-compressed relative recovery (20/10/40)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cut=P.index.max()
# Unlike zero-gated predecessor, every 20d relative loser participates.  Short/long
# volatility compression continuously amplifies (or attenuates) the recovery score.
v10=R.rolling(10,min_periods=8).std(); v40=R.rolling(40,min_periods=30).std()
mult=(1 + 0.75*np.log(v40.div(v10.replace(0,np.nan)))).clip(0.25,2.0)
raw=(-P.pct_change(20))*mult
F=raw.sub(raw.median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   s=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(s): z.append(s);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':min(ns)}
print('FACTOR continuous_volatility_compressed_relative_recovery_20_10_40 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20):print('H',h,metric(h))
for lab,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',lab,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
# Direct novelty diagnostic against the most structurally similar admitted factor:
# smooth peer-relative drawdown recovery: -(60d return), smoothed 10d, peer demeaned.
peer=(-P.pct_change(60)).rolling(10,min_periods=7).mean(); peer=peer.sub(peer.median(axis=1),axis=0).shift(1)
rhos=[]
for t in F.index:
 q=pd.concat([F.loc[t],peer.loc[t]],axis=1).dropna()
 if len(q)>=8:rhos.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('NOVELTY_SIMILAR_FACTOR mean_abs_daily_rho',round(float(np.mean(np.abs(rhos))),6),'max_abs_daily_rho',round(float(np.max(np.abs(rhos))),6),'dates',len(rhos))
print('NOTE full library novelty test required only if numerical gates and this candidate merit admission.')
