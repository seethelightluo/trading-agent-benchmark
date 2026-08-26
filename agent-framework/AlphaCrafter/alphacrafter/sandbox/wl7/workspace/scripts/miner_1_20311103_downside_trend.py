import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# downside-risk-adjusted medium trend: reward for 20d return, penalize only negative daily moves
D=R.where(R<0,0).rolling(40,min_periods=25).std()
sig=(P.pct_change(20)/D.replace(0,np.nan)).shift(1)
sig=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1).replace(0,np.nan),axis=0)
def ev(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   v=sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'); vals.append(v); rows.append((dt,v,int(ok.sum())))
 return pd.Series(vals),rows
for h in [1,5,10,20]:
 a,o=ev(h); print('h',h,'dates',len(a),'avg_n',round(np.mean([x[2] for x in o]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),5))
a,o=ev(10); print('assets',len(P.columns),'rows',len(P),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for k,(i,j) in enumerate([(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]): print('regime',k+1,round(a.iloc[i:j].mean(),8),'dates',j-i)
pd.DataFrame(o,columns=['date','ic','n']).to_csv('scripts/miner_1_20311103_downside_trend_ic_10d.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20311103_downside_trend_signal.csv',index=False)
