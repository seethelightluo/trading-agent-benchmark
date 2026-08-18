import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-05-28')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]))
P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change(); csmed=r.median(axis=1); disp=r.sub(csmed,axis=0).abs().median(axis=1)
# Continuous dispersion-weighted residual reversal; lagged to prevent lookahead.
base=-(P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0))
w=(disp/disp.rolling(60,min_periods=30).median()).clip(0.25,4.0)
f=base.mul(w,axis=0).shift(1)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q): a.append(q)
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns)),float(np.mean(f.notna().mean(axis=1)))
print('cutoff',cut.date(),'dates',len(P),'assets',len(U))
for H in [1,5,10,20]: print('H',H,run(H))
for n in [365,730,1095]: print('recent',n,'h5',run(5,max(0,len(P)-n-6)))
print('rank-turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
