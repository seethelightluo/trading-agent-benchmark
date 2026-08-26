import numpy as np, pandas as pd
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; ix='../persistent/index_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in assets},axis=1).sort_index().loc[:'2035-11-11'].ffill()
def macro(s): return pd.read_csv(f'{ix}/{s}.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
V=macro('VIX'); D=macro('DXY')
r5=C.pct_change(5); vol=C.pct_change().rolling(20).std()*np.sqrt(20); basef=-r5/vol.replace(0,np.nan)
vm=V>V.rolling(60).median(); dm=D>D.rolling(60).median()
factors={'vix_stress_reversal':basef.mul(vm,axis=0),'dxy_stress_reversal':basef.mul(dm,axis=0),'dual_stress_reversal':basef.mul(vm&dm,axis=0)}
for name,sig in factors.items():
 rows=[]
 for i in range(1,len(C)-20):
  x=sig.iloc[i-1]; y=C.iloc[i+20]/C.iloc[i]-1; q=x.notna()&y.notna()
  if q.sum()>=8 and x[q].nunique()>1 and y[q].nunique()>1: rows.append((C.index[i],x[q].corr(y[q],method='spearman'),q.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')['ic']; n=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')['n']
 ir=z.mean()/z.std(ddof=1)
 print(name,'IC',round(z.mean(),6),'ICIR',round(ir,4),'dates',len(z),'avgN',round(n.mean(),2),'coverage',round(n.mean()/15,4),'hit',round((z>0).mean(),3),'recent252',round(z.tail(252).mean(),6))
 turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean();print(' turnover',round(float(turn),4))
 if name=='vix_stress_reversal': sig.to_csv('scripts/miner_1_20351112_vix_stress_residual_reversal10_signal.csv',index_label='date')
