import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 p=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(p): px[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill().loc[:'2028-05-18']; r5=P/P.shift(5)-1; factor=-(r5.sub(r5.mean(axis=1),axis=0))
ics={h:[] for h in [1,5,10,20]}; dates={h:[] for h in ics}
for dt in factor.index:
 for h in ics:
  future=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([factor.loc[dt],future],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): ics[h].append(q); dates[h].append(dt)
for h in ics:
 x=np.array(ics[h]); print(h,'dates',len(x),'meanIC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
valid=factor.notna().sum(axis=1); used=valid>=8
print('universe',len(px),'dates',len(factor),'avg_names',round(valid[used].mean(),2),'coverage',round(valid[used].mean()/len(px),4),'min',valid[used].min())
ranks=factor.rank(axis=1,pct=True); turn=[]
for i in range(5,len(ranks)):
 if used.iloc[i] and used.iloc[i-5]: turn.append(np.nanmean(abs(ranks.iloc[i]-ranks.iloc[i-5])))
print('rank_turnover_5d',round(np.mean(turn),6))
for label,cond in [('early',factor.index<='2023-12-31'),('late',factor.index>='2024-01-01'),('recent250',factor.index>=factor.index[-251])]:
 x=np.array([ic for dt,ic in zip(dates[10],ics[10]) if cond[factor.index.get_loc(dt)]])
 print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
