import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2035-12-09')
P=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index()
P=P.loc[P.index<=cutoff]; r=P.pct_change(); mkt=r.mean(axis=1)
rel60=P.pct_change(60).sub(r.rolling(60).sum(),axis=0); vol40=r.rolling(40).std()*np.sqrt(40)
breadth=(r.rolling(20).sum()>0).mean(axis=1); gate=(breadth>=0.50).astype(float)
fac=(rel60/(vol40+1e-8)).mul(gate,axis=0).shift(1)
for H in [5,10,20]:
 rows=[]
 for i in range(1,len(P)-H):
  x=fac.iloc[i]; y=P.iloc[i+H]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: rows.append((P.index[i],ok.sum(),x[ok].corr(y[ok])))
 q=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); icir=q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252/H)
 print('H%d dates %d avgN %.2f IC %.8f ICIR %.8f hit %.4f recent252 %.8f'%(H,len(q),q.n.mean(),q.ic.mean(),icir,(q.ic>0).mean(),q.tail(252).ic.mean()))
 q['year']=q.date.dt.year; print(q.groupby(pd.cut(q.year,[2019,2024,2029,2034,2036])).ic.agg(['mean','count']).to_string())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20351210_breadth_gated_relative_trend60_signal.csv',index=False)
print('coverage %.4f turnover %.6f period %s %s'%(fac.notna().mean().mean(),np.nanmean(np.abs(np.diff(np.nan_to_num(fac.values,nan=0),axis=0))),P.index.min().date(),P.index.max().date()))
