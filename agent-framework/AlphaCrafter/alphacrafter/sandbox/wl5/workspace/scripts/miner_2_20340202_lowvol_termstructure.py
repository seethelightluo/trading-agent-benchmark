import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Volatility term-structure factor: low short-term volatility versus long-term,
# favoring assets whose recent risk has compressed; lagged and normalized cross-sectionally.
v10=r.rolling(10).std(); v60=r.rolling(60).std()
f=-(v10/v60.replace(0,np.nan)-1)
fr=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':'2034-02-01']
mean=z.ic.mean(); sd=z.ic.std(ddof=1); ir=mean/sd*np.sqrt(252)
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[z.index].mean()
print('dates',len(z),'instruments',15,'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',mean,'ICIR',ir,'hit',(z.ic>0).mean(),'turnover',turn)
for h in [5,10,20]:
 yy=p.shift(-h).div(p)-1; q=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]: print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]))
out=f.loc[z.index].stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_2_20340202_lowvol_termstructure_signal.csv',index=False);print('artifact rows',len(out))
