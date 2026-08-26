import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); q=P.pct_change(20); r=P.pct_change(); vol=r.rolling(40,min_periods=30).std()
rev=-q.sub(q.mean(axis=1),axis=0)
acc=(P.pct_change(10)-P.pct_change(20).shift(10))/vol
# cross-sectional z scores, equal interpretable blend, lagged one session
z=lambda x:x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
sig=(z(rev)+z(acc)).div(2).shift(1)
Y=P.shift(-1)/P-1; vals=[];rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&Y.loc[dt].notna()
 if v.sum()>=8:
  ic=sig.loc[dt,v].corr(Y.loc[dt,v],method='spearman');vals.append(ic);rows.append((dt,ic,int(v.sum())))
a=pd.Series(vals);print('dates',len(a),'avg_n',np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10]:
 y=P.shift(-h)/P-1;zv=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:zv.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'))
 zv=pd.Series(zv);print('h',h,'dates',len(zv),'IC %.8f ICIR %.8f'%(zv.mean(),zv.mean()/zv.std(ddof=1)))
print('coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[round(a.iloc[i:j].mean(),8) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310811_blended_reversal_accel_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310811_blended_reversal_accel_signal.csv',index=False)
