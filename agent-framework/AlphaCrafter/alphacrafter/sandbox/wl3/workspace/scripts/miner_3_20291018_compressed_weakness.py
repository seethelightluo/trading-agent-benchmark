import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index(); lr=np.log(px).diff()
ret20=np.log(px/px.shift(20)); rel=ret20.sub(ret20.median(axis=1),axis=0)
v10=lr.rolling(10).std(); v60=lr.rolling(60).std(); compression=(v10/v60).clip(0.35,2.0)
trend120=np.log(px/px.shift(120)); gate=.55+.45/(1+np.exp(-trend120/.15))
# Compressed relative-weakness reversal: favor 20d relative losers when their short volatility is compressed, softly trend-gated.
f=(-rel*compression*gate).shift(1); f=f.sub(f.median(axis=1),axis=0)
rows_by={}
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index):
  if i+h>=len(px): break
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rows_by[h]=x
 print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lab,z in [('2020_22',x.loc['2020':'2022']),('2023_25',x.loc['2023':'2025']),('2026_29',x.loc['2026':'2029']),('recent250',x.tail(250))]: print(lab,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
rr=f.rank(axis=1,pct=True); print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(D))),'turnover %.4f'%rr.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20291018_compressed_weakness_signal.csv',index=False)
