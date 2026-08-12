import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=get_account_dict()['watch_list']
def px(s):
 try:d=get_stock_daily_data(s,days=5000)
 except Exception:d=get_index_daily_data(s,days=5000)
 z=d[['date','close']].copy();z.date=pd.to_datetime(z.date);return z.set_index('date').close
P=pd.concat({s:px(s) for s in U},axis=1).sort_index().ffill();v=get_index_daily_data('VIX',days=5000);V=v.set_index(pd.to_datetime(v.date)).close.reindex(P.index).ffill()
r=np.log(P).diff();m=P.pct_change(20);vol=r.rolling(40).std()*np.sqrt(252);vr=V.shift(1).rolling(252,min_periods=60).rank(pct=True);g=(vr-.5).clip(-.5,.5)*2
f=((m/vol).shift(1)*(1-g)+(-m/vol).shift(1)*g).replace([np.inf,-np.inf],np.nan)
def ics(h):
 fr=P.pct_change(h).shift(-h);rr=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:rr.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 return pd.Series(rr).dropna()
print('dates',sum(len(ics(h)) for h in [1]) ,'avg_n',len(U),'coverage',f.notna().sum().sum()/(len(f)*len(U)))
for h in [1,5,10,20]:
 q=ics(h);print(h,'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20310220_regime_blended_trend_signal.csv',index=False)
