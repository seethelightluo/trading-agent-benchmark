import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
ret20=P.pct_change(20).shift(1)
down=r.where(r<0).rolling(40,min_periods=20).std().shift(1)
vol=r.rolling(20,min_periods=15).std().shift(1)
sig=ret20/(down*np.sqrt(40)+1e-12)*(vol.rolling(60,min_periods=30).mean().shift(1)/(vol+1e-12)).clip(.5,2)
sig=sig.sub(sig.median(axis=1),axis=0)
y=P.shift(-1)/P-1
ics=[]; rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8:
  q=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); ics.append(q); rows.append((dt,q,int(v.sum())))
a=pd.Series(ics)
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]))
print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; b=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&yy.loc[dt].notna()
  if v.sum()>=8:b.append(sig.loc[dt,v].corr(yy.loc[dt,v],method='spearman'))
 b=pd.Series(b);print('h',h,'dates',len(b),'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20310505_quality_trend_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310505_quality_trend_signal.csv',index=False)
