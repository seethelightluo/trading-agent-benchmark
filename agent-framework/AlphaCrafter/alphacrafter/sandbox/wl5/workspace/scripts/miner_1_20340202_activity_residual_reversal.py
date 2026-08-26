import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in U}
close=pd.DataFrame({a:x.close for a,x in px.items()}).sort_index()
vol=pd.DataFrame({a:x.volume for a,x in px.items()}).reindex(close.index)
r=close.pct_change(); ret5=r.rolling(5).sum(); resid=ret5.sub(ret5.median(axis=1),axis=0)
v20=r.rolling(20).std(); volratio=np.log((vol.rolling(20).mean()/vol.rolling(120).mean()).replace(0,np.nan)).clip(-2,2)
# high activity amplifies the causal short-term residual reversal
f=-resid/v20.replace(0,np.nan)*(1+0.30*volratio)
fr=close.shift(-10).div(close)-1
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-07-01':'2034-02-01']
mean=z.ic.mean();sd=z.ic.std(ddof=1);print('dates',len(z),'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(z.ic>0).mean())
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean(axis=1).reindex(z.index).mean())
for h in [5,20]:
 yy=close.shift(-h).div(close)-1;q=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]:print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]))
out=f.loc[z.index].stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_1_20340202_activity_residual_reversal_signal.csv',index=False);print('artifact rows',len(out))
