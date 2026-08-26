import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Volatility-shock conditioned residual reversal, all inputs lagged through date d
ret5=r.rolling(5).sum(); residual=ret5.sub(ret5.median(axis=1),axis=0)
vol20=r.rolling(20).std(); vol120=r.rolling(120).std()
shock=(vol20/vol120.replace(0,np.nan)-1).clip(-1,2)
f=-residual/vol20.replace(0,np.nan)*(1+0.40*shock)
fr=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-07-01':'2034-01-18']
mean=z.ic.mean(); sd=z.ic.std(ddof=1); ir=mean/sd*np.sqrt(252)
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).loc[z.index].mean()
print('dates',len(z),'instruments',15,'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',mean,'ICIR',ir,'hit',(z.ic>0).mean(),'turnover',turn)
for h in [5,10,20]:
 yy=p.shift(-h).div(p)-1; q=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]: print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]))
out=f.loc[z.index].stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_1_20340119_volshock_residual_reversal_signal.csv',index=False);print('artifact rows',len(out))
