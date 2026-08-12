import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}; P=pd.DataFrame(px).sort_index(); R=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
rv=R.rolling(20).std(); raw=-R.rolling(5).sum()/rv.replace(0,np.nan); stress=(vix>vix.rolling(60).quantile(.70)) | (dxy.pct_change(20)>0); fac=raw.where(stress,np.nan).shift(1)
rows=[]; turnover=[]
for i,t in enumerate(P.index):
 if i+10>=len(P): continue
 z=pd.concat([fac.loc[t],(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:
  rows.append((t,spearmanr(z.iloc[:,0],z.y).statistic,len(z)))
  if i>0:
   old=fac.iloc[i-1].dropna(); new=fac.loc[t].dropna(); c=old.index.intersection(new.index)
   if len(c)>=8: turnover.append(new[c].rank().sub(old[c].rank()).abs().mean()/len(c))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'avg_n',out.n.mean(),'active_fraction',len(out)/len(P.index)); print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1),(out.ic>0).mean(),np.nanmean(turnover)))
for label,sl in [('2020-23',out.loc[:'2023']),('2024-26',out.loc['2024':'2026']),('2027-29',out.loc['2027':'2029']),('2030-32',out.loc['2030':'2032'])]:
 if len(sl): print(label,len(sl),'IC %.5f ICIR %.5f hit %.3f'%(sl.ic.mean(),sl.ic.mean()/sl.ic.std(ddof=1),(sl.ic>0).mean()))
for h in [5,10,20]:
 vals=[]
 for i,t in enumerate(P.index):
  if i+h>=len(P): continue
  z=pd.concat([fac.loc[t],(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.y).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
fac.to_csv('scripts/miner_1_20320902_macro_stress_reversal_signal.csv')
