import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-05-14')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change()
# Volatility compression: assets whose recent 10d realized volatility is low relative to
# their 60d volatility are favored. Signal is lagged one completed session.
v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=40).std()
f=-(v10/v60.replace(0,np.nan)).shift(1)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns)),float(np.mean(f.notna().mean(axis=1)))
print('cutoff',cut.date(),'dates',len(P),'assets',len(U))
for H in [1,5,10,20]: print('H',H,run(H))
for n in [365,730,1095]: print('recent',n,'h10',run(10,max(0,len(P)-n-11)))
print('rank-turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
