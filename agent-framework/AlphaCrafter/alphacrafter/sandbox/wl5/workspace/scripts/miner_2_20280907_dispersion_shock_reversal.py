import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:'2028-09-06']; r=p.pct_change()
# cross-sectional dispersion shock, followed by 5d reversal
csdisp=r.std(axis=1); disp=csdisp.rolling(5).mean()
shock=(disp>disp.rolling(60,min_periods=30).quantile(.70)).astype(float)
factor=-r.rolling(5).sum().mul(shock,axis=0).fillna(0)
rows=[]
for i in range(60,len(p)-10):
 vals=factor.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=vals.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],spearmanr(vals[ok],fr[ok]).statistic,ok.mean(),shock.iloc[i]))
x=pd.DataFrame(rows,columns=['date','ic','coverage','shock']).set_index('date')
for label,z in [('all',x),('2020-25',x.loc[:'2025-12-31']),('2026',x.loc['2026-01-01':'2026-12-31']),('2027',x.loc['2027-01-01':'2027-12-31']),('2028',x.loc['2028-01-01':])]:
 ic=z.ic.dropna(); print(label,'dates',len(z),'meanIC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(z.coverage.mean(),4),'shockdays',round(z.shock.mean(),4))
q=factor.rank(axis=1,pct=True); turn=q.diff().abs().mean(axis=1).dropna(); print('turnover',turn.loc[x.index].mean()); print('range',x.index.min(),x.index.max())
for h in [1,5,10,20]:
 a=[]
 for i in range(60,len(p)-h):
  ok=factor.iloc[i].notna()&p.iloc[i+h].notna()
  if ok.sum()>=8:a.append(spearmanr(factor.iloc[i][ok],(p.iloc[i+h]/p.iloc[i]-1)[ok]).statistic)
 print('h',h,'IC',round(np.nanmean(a),6),'n',len(a))
