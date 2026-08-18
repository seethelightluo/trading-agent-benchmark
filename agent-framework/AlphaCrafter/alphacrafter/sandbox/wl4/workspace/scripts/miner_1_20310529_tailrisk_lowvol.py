import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-05-28')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}; dates=sorted(set.intersection(*[set(x.index) for x in px.values()]));P=pd.DataFrame({s:px[s].reindex(dates) for s in U});r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); down=(r.clip(upper=0)**2).rolling(20,min_periods=15).mean().pow(.5);up=(r.clip(lower=0)**2).rolling(20,min_periods=15).mean().pow(.5);f=(1/vol/(1+down/(up+1e-8))).shift(1)
def run(H,start=0):
 a=[];ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i];y=P.iloc[i+1+H]/P.iloc[i+1]-1;ok=x.notna()&y.notna();ns.append(ok.sum())
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:a.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.array(a);return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns),f.notna().mean().mean()
print('dates',len(P),'assets',15)
for h in [1,5,10,20]:print('H',h,run(h))
for n in [365,730,1095]:print('recent',n,'H10',run(10,len(P)-n-11))
print('rank-turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean()))
