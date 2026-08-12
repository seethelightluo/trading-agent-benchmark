import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index(); lr=np.log(px).diff(); r5=np.log(px/px.shift(5)); v20=lr.rolling(20).std()*np.sqrt(5)
# Downside-adjusted short reversal: recent losers receive extra weight when their path had negative daily skew.
down=lr.clip(upper=0).rolling(5).sum().abs(); total=lr.abs().rolling(5).sum(); downside=(down/(total+1e-9)).clip(0,1)
rel=r5.sub(r5.median(axis=1),axis=0); f=(-rel/(v20+1e-9)*(0.5+downside)).shift(1); f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index):
  if i+h>=len(px):break
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 print('recent250',len(x.tail(250)),'IC %.6f ICIR %.6f'%(x.tail(250).ic.mean(),x.tail(250).ic.mean()/x.tail(250).ic.std()))
rr=f.rank(axis=1,pct=True);print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(D))),'turnover %.4f'%rr.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20291018_downside_reversal_signal.csv',index=False)
