import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: downside-risk asymmetry. Higher factor = less downside volatility relative to total volatility.
# All inputs are lagged one session before forward return.
series={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<80: continue
 d=d[['date','close']].copy(); d['r']=d.close.pct_change(); series[s]=d.set_index('date')
px=pd.DataFrame({s:v.close for s,v in series.items()}); rets=px.pct_change()
# factor at t: negative downside semideviation / total volatility, 20d; lower downside asymmetry preferred
# factor is negative ratio, so larger means better.
down=rets.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
tot=rets.rolling(20,min_periods=15).std()
f=-(down/tot)
rows=[]
for h in [1,5,10]:
  ics=[]; nms=[]
  fr=px.pct_change(h).shift(-h)
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); nms.append(len(z))
  x=pd.Series(ics).dropna(); print('horizon',h,'dates',len(x),'avg_n',round(np.mean(nms),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(f.notna().sum().sum()/(f.shape[0]*len(U)),4))
  print('regime', {str(y):round(x[[d.year==y for d in x.index]].mean(),5) for y in []})
# turnover based on daily rank ordering
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('assets',len(series),'dates',len(f),'turnover',round(turn,5),'last',f.iloc[-1].dropna().round(4).to_dict())
# persist signal artifact for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_downside_asymmetry_signal.csv',index=False)
