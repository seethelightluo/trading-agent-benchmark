import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-08-25')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut]
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change(); hi=pd.DataFrame({s:D[s].high for s in U}).reindex(p.index); lo=pd.DataFrame({s:D[s].low for s in U}).reindex(p.index)
# Strong one-day shock reversal: yesterday's loss scaled by its intraday range, all based on completed day.
fac=(-r.shift(1)*(hi.shift(1)-lo.shift(1))/p.shift(1)).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print(h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
 for loY,hiY in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[loY<=d.year<=hiY for d in ds]]; print('reg',loY,hiY,len(q),round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(fac.notna().mean().mean(),4),'turn',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270826_range_shock_signal.csv',index=False)
