import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 if d is not None:p[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(p).sort_index();r=np.log(P).diff(); ret=P/P.shift(10)-1
neg=r.where(r<0); dv=np.sqrt((neg**2).rolling(30,min_periods=10).mean()).replace(0,np.nan)
# Contrarian ten-day return, scaled by downside risk, all information lagged one day.
f=(-ret/dv).shift(1);Y={h:P.shift(-h)/P-1 for h in [1,5,10,20]};rows=[];sig=[]
for dt in f.index:
 x=f.loc[dt];y=Y[10].loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()));sig.append((dt,*x.reindex(U).tolist()))
I=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');v=I.ic
print('factor=downside_reversal dates',len(I),'avgN',I.n.mean(),'coverage',I.n.mean()/15);print('IC10',v.mean(),'ICIR',v.mean()/v.std(),'hit',(v>0).mean(),'recent365',v.tail(365).mean()/v.tail(365).std())
for h,Yh in Y.items():
 a=[]
 for dt in f.index:
  x=f.loc[dt];y=Yh.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append(x[ok].corr(y[ok],method='spearman'))
 print('decay',h,np.nanmean(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());I.to_csv('scripts/miner_2_20350104_downside_reversal_ic.csv');pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20350104_downside_reversal_signal.csv')
