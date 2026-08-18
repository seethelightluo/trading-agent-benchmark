import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-11-27')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut]
 px[s]=d['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change()
# Contrarian 10-session move, normalized by trailing downside risk; all inputs lagged one completed session.
down=r.where(r<0).rolling(30,min_periods=10).std()
f=(-r.rolling(10,min_periods=10).sum().div(down.replace(0,np.nan))).shift(1)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns))
for H in [1,5,10,20]: print('H',H,run(H))
for n in [260,520]: print('recent',n,'H20',run(20,max(0,len(P)-n-21)))
print('dates',len(P),'assets',len(U),'coverage',float(f.notna().mean().mean()),'turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
