import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); V=R.rolling(40,min_periods=30).std()
def csz(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
# Candidate: medium-horizon reversal divided by trailing volatility, lagged one day.
# Volatility normalization prevents crypto/commodity scale domination and should reduce signal churn.
sig=csz(-P.pct_change(20)/V).shift(1)
def evaluate(h):
 y=P.shift(-h)/P-1; vals=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append((dt,sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'),int(ok.sum())))
 a=pd.Series([x[1] for x in vals]); return a,vals
for h in [1,5,10,20]:
 a,o=evaluate(h); print('h',h,'dates',len(a),'avg_n',round(np.mean([x[2] for x in o]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),5))
a,o=evaluate(1)
print('history_dates',len(P),'assets',len(P.columns),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for k,(i,j) in enumerate([(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]): print('regime',k+1,round(a.iloc[i:j].mean(),8), 'dates',j-i)
pd.DataFrame(o,columns=['date','ic','n']).to_csv('scripts/miner_3_20311020_volnorm_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311020_volnorm_reversal_signal.csv',index=False)
