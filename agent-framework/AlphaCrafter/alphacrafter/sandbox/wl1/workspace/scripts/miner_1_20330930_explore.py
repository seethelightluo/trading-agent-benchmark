import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# candidates, all lagged naturally by evaluating signal at t against t+1..t+h
signals={}
s20=p.pct_change(20); s60=p.pct_change(60); down=r.clip(upper=0).rolling(30).std(); vol=r.rolling(30).std()
signals['downside_resilient_rel30']=p.pct_change(30)/down
signals['acceleration_downside']= (s20-s60)/down
signals['trend_quality_20_60']=s20/vol + .5*s60/r.rolling(60).std()
for name,x in signals.items():
  # cross sectional rank; raw factor IC equivalent rank correlation
  vals=[]
  for h in [1,5,10,20]:
    rows=[]
    for i in range(len(p)-h):
      dt=p.index[i]; nxt=p.iloc[i+h]/p.iloc[i]-1
      a=x.iloc[i]; z=pd.concat([a,nxt],axis=1).dropna()
      if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=pd.Series(rows).dropna(); mu=q.mean(); sd=q.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd else np.nan
    vals.append((h,len(q),mu,icir,(q>0).mean()))
  # turnover rank changes daily
  ranks=x.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).mean()
  print(name,'metrics',vals,'turn',turn,'coverage',x.notna().sum(axis=1).mean()/15)
  # save signal artifact
  out=x.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20330930_'+name+'_signal.csv')
  for label,sl in [('early',slice('2020','2027')),('mid',slice('2028','2030')),('recent',slice('2031','2033'))]:
    q=[]
    for i in range(len(p)-10):
      if not (p.index[i]>=pd.Timestamp(sl.start) and p.index[i]<=pd.Timestamp(sl.stop+'-12-31')): continue
      z=pd.concat([x.iloc[i],(p.iloc[i+10]/p.iloc[i]-1)],axis=1).dropna()
      if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=pd.Series(q); print(' ',label,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
