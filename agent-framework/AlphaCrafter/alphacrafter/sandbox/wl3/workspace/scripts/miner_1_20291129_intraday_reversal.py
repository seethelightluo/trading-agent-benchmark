import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];O={};C={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  q=d.set_index('date');O[s]=q.open.astype(float);C[s]=q.close.astype(float)
opx=pd.DataFrame(O).sort_index();px=pd.DataFrame(C).reindex(opx.index);lr=np.log(px).diff();v20=lr.rolling(20).std()*np.sqrt(20)
intra=np.log(px/opx);z=intra/(v20+1e-9)
# Reversal of abnormal intraday close-open displacement, lagged one session.
f=(-z).shift(1);f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index):
  if i+h>=len(px):break
  q=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(q)>=8:rows.append((dt,len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lab,q in [('2026_27',x.loc['2026':'2027']),('2028_29',x.loc['2028':'2029']),('recent250',x.tail(250))]:print(lab,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()))
rr=f.rank(axis=1,pct=True);print('dates',len(px),'instruments',len(C),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(C))),'turnover %.4f'%(rr.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20291129_intraday_reversal_signal.csv',index=False)
