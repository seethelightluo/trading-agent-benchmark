import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 return x[['date','close']].drop_duplicates('date').set_index('date')['close'] if x is not None and len(x) else None
C=pd.DataFrame({s:load(s) for s in U}).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-11-07')]; r=C.pct_change()
spread=C.US10Y-C.CN10Y; shock=(spread-spread.shift(10)).abs(); gate=(shock>shock.rolling(252,min_periods=126).quantile(.65)).astype(float)
q=r.rolling(5).sum(); base=-(q.sub(q.mean(axis=1),axis=0))/(r.rolling(20).std()*np.sqrt(5)+1e-8); sig=base.mul(1+.75*gate,axis=0)
fw=C.shift(-20)/C-1; X=sig.rank(axis=1); Y=fw.rank(axis=1); xm=X.mean(axis=1); ym=Y.mean(axis=1)
num=X.sub(xm,axis=0).mul(Y.sub(ym,axis=0)).sum(axis=1); den=np.sqrt(X.sub(xm,axis=0).pow(2).sum(axis=1)*Y.sub(ym,axis=0).pow(2).sum(axis=1)); ic=(num/den).dropna()
print('h=20 dates=%d avg_inst=%.3f IC=%.8f ICIR=%.8f hit=%.4f'%(len(ic),sig.notna().sum(axis=1).mean(),ic.mean(),ic.mean()/ic.std(ddof=1)*np.sqrt(len(ic)),(ic>0).mean()))
print('coverage=%.6f turnover=%.6f shock_frequency=%.4f instruments=15 dates=%d end=%s'%(sig.notna().sum().sum()/(len(sig)*15),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),gate.mean(),len(sig),C.index.max().date()))
