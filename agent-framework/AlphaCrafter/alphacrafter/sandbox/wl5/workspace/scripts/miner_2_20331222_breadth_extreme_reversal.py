import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index()
r=px.pct_change()
breadth=r.gt(0).rolling(20,min_periods=8).mean().mean(axis=1)
raw=-px.pct_change(5)
gate=0.5+2*(breadth-0.5).abs()
f=raw.mul(gate,axis=0); fwd=px.shift(-10)/px-1
rows=[]; turns=[]; prev=None
for dt in px.index:
 a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append((dt,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
  if prev is not None:
   common=ok&prev.notna(); turns.append((a[common].rank(pct=True)-prev[common].rank(pct=True)).abs().mean())
  prev=a
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2026-07-16':'2033-12-21']
mean=z.ic.mean(); sd=z.ic.std(ddof=1)
print('dates',len(z),'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',mean,'ICIR_ann',mean/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'turnover',np.nanmean(turns))
for h in [5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]
 for dt in z.index:
  a=f.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
for a,b in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-12-21')]:
 q=z.loc[a:b]; print('regime',a,b,len(q),q.ic.mean(),q.ic.std(ddof=1),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252))
f.loc[z.index].to_csv('scripts/miner_2_20331222_breadth_extreme_reversal_signal.csv',index_label='date')
