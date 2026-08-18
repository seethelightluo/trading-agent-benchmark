import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-04-16')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}; dates=sorted(set.intersection(*[set(x.index) for x in px.values()]));P=pd.DataFrame({s:px[s].reindex(dates) for s in U});r=P.pct_change()
# One-day cross-sectional residual reversal, lagged one day.
f=-(r.sub(r.median(axis=1),axis=0)).shift(1)
def run(H,start=0):
 a=[];ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i];y=P.iloc[i+1+H]/P.iloc[i+1]-1;ok=x.notna()&y.notna();ns.append(ok.sum())
  if ok.sum()>=8:a.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.array(a);return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns),f.notna().mean(axis=1).mean()
print('cutoff',cut.date(),'dates',len(P),'assets',len(U))
for h in [1,5,10,20]:print('H',h,run(h))
for n in [365,730,1095]:print('recent',n,run(5,max(0,len(P)-n-6)))
print('turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>.1).mean()))
