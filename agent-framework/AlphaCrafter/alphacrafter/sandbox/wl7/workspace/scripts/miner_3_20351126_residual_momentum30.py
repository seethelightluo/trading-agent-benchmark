import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
P=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index()
r=P.pct_change(); mkt=r.mean(axis=1)
excess=P.pct_change(30).sub(mkt.rolling(30).sum(),axis=0)
raw=excess/(r.sub(mkt,axis=0).rolling(40).std()*np.sqrt(30)+1e-8)
fac=(-raw).shift(1)
rows=[]
for i in range(1,len(P)-10):
 x=fac.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; qv=x.notna()&y.notna()
 if qv.sum()>=8: rows.append((P.index[i],qv.sum(),x[qv].corr(y[qv])))
q=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
icir=q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252)
turn=np.nanmean(np.abs(np.diff(np.nan_to_num(fac.values,nan=0),axis=0)))
print('dates',len(q),'assets',P.shape[1],'period',P.index.min().date(),P.index.max().date(),'avgN',q.n.mean())
print('H10 IC %.8f ICIR %.8f hit %.4f recent252 %.8f coverage %.4f turnover %.6f'%(q.ic.mean(),icir,(q.ic>0).mean(),q.tail(252).ic.mean(),fac.notna().mean().mean(),turn))
q['year']=q.date.dt.year
print(q.groupby(pd.cut(q.year,[2019,2024,2029,2034,2036])).ic.agg(['mean','count']).to_string())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20351126_residual_momentum30_signal.csv',index=False)
