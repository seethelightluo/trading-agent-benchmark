import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro={'DXY','USDCNY','USDJPY','EURUSD','VIX'}
def get(sym):
    d=get_stock_daily_data(sym,days=4000)
    if d is None or len(d)<100: d=get_index_daily_data(sym,days=4000)
    return d
# Candidate: downside-asymmetry adjusted medium momentum. Favor positive 30d return with low downside semideviation.
series={}
for s in U:
 d=get(s)
 if d is not None:
  x=d[['date','close']].copy(); x['r']=x.close.pct_change(); series[s]=x.set_index('date')
px=pd.DataFrame({s:v.close for s,v in series.items()}).sort_index(); re=px.pct_change()
# lagged signal: at date t use data through t, forward returns t+1..t+h
pos=re.clip(lower=0).rolling(30,min_periods=20).mean()
down=(-re.clip(upper=0)).rolling(30,min_periods=20).mean()
vol=re.rolling(30,min_periods=20).std()
f=(re.rolling(30,min_periods=20).sum()/vol.replace(0,np.nan)) * (pos/(down+pos+1e-8))
f=f.shift(1)
rows=[]
for h in [1,3,5,10]:
  fr=px.shift(-h)/px-1
  ics=[]; nobs=[]; turns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); nobs.append(len(z))
  a=pd.Series(ics).dropna(); ic=a.mean(); icir=ic/a.std(ddof=1)*np.sqrt(len(a))
  print('H',h,'IC',round(ic,7),'ICIR',round(icir,4),'hit',round((a>0).mean(),4),'dates',len(a),'avgN',round(np.mean(nobs),2))
# coverage and turnover (rank changes daily)
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/len(U),'avg_valid',valid.mean())
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
# save artifact at admitted horizon
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280420_asym_mom_signal.csv',index=False)
print('artifact rows',len(out),'dates',len(f))
