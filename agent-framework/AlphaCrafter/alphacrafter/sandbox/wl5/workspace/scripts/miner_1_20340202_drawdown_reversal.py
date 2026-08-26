import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in U}).sort_index();r=p.pct_change()
# Drawdown-oversold recovery: cross-sectional residual reversal, strengthened for assets near 60d lows.
ret10=r.rolling(10).sum();res=ret10-ret10.median(axis=1).values[:,None]; dd=p/p.rolling(60).max()-1
f=-res*(1+0.6*(-dd).clip(0,0.5)); fr=p.shift(-10)/p-1
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-07-01':'2034-02-01'];m=z.ic.mean();sd=z.ic.std(ddof=1)
print('dates',len(z),'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',m,'ICIR',m/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex(z.index).mean())
for h in [5,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]:print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]))
f.loc[z.index].stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20340202_drawdown_reversal_signal.csv',index=False)
