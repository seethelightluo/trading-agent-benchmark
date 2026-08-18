import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-10-02')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); px[s]=d.close.loc[:cut]
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.loc[:cut]
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])&set(dxy.index)); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); D=dxy.reindex(dates)
# Stress is a positive 20-day DXY impulse relative to its trailing 120-day distribution; contrarian asset momentum is activated in stress.
dret=D.pct_change(20); scale=D.pct_change().rolling(120,min_periods=60).std()*np.sqrt(20); stress=(dret/scale).clip(lower=0).clip(upper=3)
f=(-P.pct_change(20)).mul(stress,axis=0)
def run(H,start=0):
 a=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(ok.sum())
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): a.append(z)
 a=np.array(a); return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns)
for H in [1,5,10,20]: print('H',H,tuple(round(x,6) if isinstance(x,float) else x for x in run(H)))
print('recent365_h20',tuple(round(x,6) if isinstance(x,float) else x for x in run(20,max(0,len(P)-365-21))))
print('dates',len(P),'assets',len(U),'coverage',round(f.notna().mean().mean(),4),'active',round((stress>0).mean(),4))
