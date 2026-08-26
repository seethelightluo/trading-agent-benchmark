import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:d.date=pd.to_datetime(d.date);raw[s]=d.set_index('date').close
P=pd.DataFrame(raw).sort_index();r=P.pct_change();res=r.sub(r.median(axis=1),axis=0);v=r.rolling(20,min_periods=10).std()
# Three-day volatility-normalized idiosyncratic reversal, activated in the lower-dispersion regime.
shock=res.rolling(3,min_periods=3).sum()/(v*np.sqrt(3)+1e-9);disp=res.std(axis=1); gate=disp<disp.shift(1).rolling(120,min_periods=40).quantile(.40);sig=(-shock.where(gate,0)).shift(1);y=P.shift(-1)/P-1
rows=[]
for dt in sig.index:
 q=sig.loc[dt].notna()&y.loc[dt].notna()
 if q.sum()>=8:rows.append((dt,sig.loc[dt,q].corr(y.loc[dt,q],method='spearman'),int(q.sum())))
a=pd.Series([z[1] for z in rows]);n=len(a);print('dates',n,'assets',len(P.columns),'avg_n %.2f'%np.mean([z[2] for z in rows]));print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1;z=[]
 for dt in sig.index:
  q=sig.loc[dt].notna()&yy.loc[dt].notna()
  if q.sum()>=8:z.append(sig.loc[dt,q].corr(yy.loc[dt,q],method='spearman'))
 z=pd.Series(z);print('h',h,'dates',len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()));print('regimes',*[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
out='scripts/miner_1_20310714_lowdisp_3d_reversal';pd.DataFrame(rows,columns=['date','ic','n']).to_csv(out+'_ic.csv',index=False);sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv(out+'_signal.csv',index=False);print('signal_artifact='+out+'_signal.csv')
