import numpy as np, pandas as pd
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; ix='../persistent/index_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in assets},axis=1).sort_index().loc[:'2035-10-28']
def macro(s): return pd.read_csv(f'{ix}/{s}.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
V=macro('VIX'); D=macro('DXY')
r5=C.pct_change(5); vol=C.pct_change().rolling(20).std()*np.sqrt(20); basef=r5/vol.replace(0,np.nan)
stress=((V>V.rolling(60).median()) | (D>D.rolling(60).median())).astype(float)
sig=basef.mul(1-2*stress,axis=0)
rows=[]
for i in range(1,len(C)-20):
 x=sig.iloc[i-1].values; ok=np.isfinite(x)
 if ok.sum()<8:continue
 for h in [1,5,10,20]:
  y=C.iloc[i+h].values/C.iloc[i].values-1;q=ok&np.isfinite(y)
  if q.sum()>=8: rows.append((C.index[i],h,np.corrcoef(x[q],y[q])[0,1],q.sum()))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('rows',len(df),'dates',df.date.nunique(),'assets',C.shape[1],'period',C.index.min(),C.index.max())
for h in [1,5,10,20]:
 z=df[df.h==h].dropna();print(h,'IC %.8f ICIR %.8f hit %.4f nobs %d avgN %.2f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252),(z.ic>0).mean(),len(z),z.n.mean()))
print('coverage %.6f turnover %.6f'%(np.isfinite(sig).sum().sum()/sig.size,np.nanmean(np.abs(np.diff(np.nan_to_num(sig.values,nan=0),axis=0)))))
z=df[df.h==10].dropna()
for lo,hi in [('2030-01-01','2034-12-31'),('2035-01-01','2035-10-28'),('2034-10-01','2035-10-28')]:
 a=z[(z.date>=lo)&(z.date<=hi)];print('regime',lo,hi,len(a),a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(252) if len(a)>2 else np.nan)
out=sig;out.index.name='date';out.to_csv('scripts/miner_3_20351029_stress_reversal_signal.csv')
