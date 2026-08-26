import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
# Shock exhaustion: short return reversal, scaled by current volatility and gated to volatility expansion relative to its own baseline.
v=r.rolling(20,min_periods=12).std().shift(1); base=v.rolling(60,min_periods=30).median().shift(1)
sig=-(r.rolling(3,min_periods=3).sum().shift(1))/(v*np.sqrt(3)+1e-12)*(v/(base+1e-12)).clip(.5,2)
sig=sig.sub(sig.median(axis=1),axis=0)
y=P.shift(-1)/P-1; rows=[]; vals=[]
for dt in sig.index:
 ok=sig.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8: rows.append((dt,sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'),int(ok.sum()))); vals.append(rows[-1][1])
a=pd.Series(vals)
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n',np.mean([x[2] for x in rows]))
print('daily IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1;b=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:b.append(sig.loc[dt,ok].corr(yy.loc[dt,ok],method='spearman'))
 b=pd.Series(b);print('h',h,'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20310505_shock_exhaustion_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310505_shock_exhaustion_signal.csv',index=False)
