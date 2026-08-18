import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-11-13')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); px[s]=d.close.loc[:cut]
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change()
# A stable continuation signal: average standardized 20d and 60d returns, requiring agreement; lagged one completed session.
rv=r.rolling(30,min_periods=20).std()*np.sqrt(30)
m20=P.pct_change(20).div(rv); m60=P.pct_change(60).div(rv)
f=((m20+m60)/2 * (np.sign(m20)*np.sign(m60)).clip(lower=0)).shift(1)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns))
for H in [1,5,10,20]: print('H',H,run(H))
print('recent365_h10',run(10,max(0,len(P)-365-11)))
print('dates',len(P),'assets',len(U),'coverage',float(f.notna().mean().mean()),'turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
