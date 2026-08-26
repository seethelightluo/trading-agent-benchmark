import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:
  z=d[['date','close']].copy();z.date=pd.to_datetime(z.date);px[s]=z.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); m=r.median(axis=1)
# Residual short shock: remove contemporaneous common cross-asset move, then fade idiosyncratic 3d move.
res=r.sub(m,axis=0); scale=res.rolling(20).std().shift(1)*np.sqrt(3)
sig=(-res.rolling(3).sum().shift(1)/scale.replace(0,np.nan)); sig=sig.sub(sig.median(axis=1),axis=0)
def test(h):
 y=P.shift(-h)/P-1;a=[];rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   q=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman');a.append(q);rows.append((dt,q,int(v.sum())))
 a=pd.Series(a);return a,rows
for h in [1,5,10,20]:
 a,rr=test(h);print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rr]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rr=test(1);print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()));print('regimes',[a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
pd.DataFrame(rr,columns=['date','ic','n']).to_csv('scripts/miner_3_20310310_residual_shock_reversal_ic.csv',index=False);sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310310_residual_shock_reversal_signal.csv',index=False)
