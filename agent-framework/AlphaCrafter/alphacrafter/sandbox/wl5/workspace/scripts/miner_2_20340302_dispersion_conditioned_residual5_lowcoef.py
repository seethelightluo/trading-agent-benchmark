import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in U}).sort_index()
r=p.pct_change(); ret5=r.rolling(5).sum(); vol20=r.rolling(20).std()
res=-ret5.sub(ret5.median(axis=1),axis=0)/vol20.replace(0,np.nan)
disp=ret5.std(axis=1); base=disp.rolling(60,min_periods=30).median(); g=(disp/base-1).clip(-1,2)
f=res.mul(1+0.20*g,axis=0).replace([np.inf,-np.inf],np.nan)
def ics(h):
 fr=p.shift(-h).div(p)-1; out=[]
 for d in f.index:
  a=f.loc[d].to_numpy(); b=fr.loc[d].to_numpy(); ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
z=ics(10).loc['2020-01-01':'2034-03-01']; mean=z.ic.mean(); ir=mean/z.ic.std(ddof=1)*np.sqrt(252)
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[z.index].mean()
print('dates',len(z),'instruments',15,'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC10',mean,'ICIR',ir,'hit',(z.ic>0).mean(),'turnover',turn,flush=True)
for h in [5,20]:
 q=ics(h).loc[z.index].ic
 print('decay',h,q.mean(),len(q),flush=True)
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]: print('regime',s,e,z.loc[s:e].ic.mean(),len(z.loc[s:e]),flush=True)
out=f.loc[z.index].stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_2_20340302_dispersion_conditioned_residual5_lowcoef_signal.csv',index=False);print('artifact rows',len(out),flush=True)
