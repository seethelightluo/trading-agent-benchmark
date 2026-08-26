import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);px[s]=x.set_index('date').close
P=pd.DataFrame(px).sort_index(); ret=P.pct_change()
# Novel candidate: reversal after an unusually compressed 10d path, scaled by recent risk.
# Compression is low short/long volatility; signal fades the recent 5d move, activated only in bottom-half compression cross-section.
rv10=ret.rolling(10).std().shift(1); rv40=ret.rolling(40).std().shift(1)
compression=rv10/rv40.replace(0,np.nan)
shock=ret.rolling(5).sum().shift(1)/ (rv20:=ret.rolling(20).std().shift(1)*np.sqrt(5)).replace(0,np.nan)
cscomp=compression.rank(axis=1,pct=True)
sig=-shock*(cscomp<0.5)
sig=sig.sub(sig.median(axis=1),axis=0)

def test(h):
 y=P.shift(-h)/P-1; vals=[]; ns=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   ic=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman');vals.append(ic);ns.append(v.sum());rows.append((dt,ic,int(v.sum())))
 a=pd.Series(vals);return a,rows
for h in [1,5,10,20]:
 a,rows=test(h);print('h',h,'dates',len(a),'avg_n',np.mean([r[2] for r in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',[a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20310224_compressed_shock_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310224_compressed_shock_reversal_signal.csv',index=False)
