import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); q=P.pct_change(30); disp=q.std(axis=1); threshold=disp.rolling(90,min_periods=60).quantile(.65)
base=-q.sub(q.mean(axis=1),axis=0); sig=base.where(disp.gt(threshold)).shift(1)
rows=[]
for h in [1,5,10]:
 y=P.shift(-h)/P-1; vals=[]; rr=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: rr.append((dt,sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'),int(ok.sum())))
 a=pd.Series([x[1] for x in rr]); print('h',h,'dates',len(a),'avg_n',round(np.mean([x[2] for x in rr]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),5))
 if h==1: rows=rr
A=pd.Series([x[1] for x in rows]); print('history_dates',len(P),'assets',len(P.columns),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6)); print('regimes',[round(A.iloc[i:j].mean(),8) for i,j in [(0,len(A)//3),(len(A)//3,2*len(A)//3),(2*len(A)//3,len(A))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20311103_dispersion_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311103_dispersion_reversal_signal.csv',index=False)
