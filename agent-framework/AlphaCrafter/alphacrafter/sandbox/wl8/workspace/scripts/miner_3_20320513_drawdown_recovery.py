import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-05-01'); p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(30,min_periods=30).std().shift(1)
prior_min=p.rolling(20,min_periods=20).min().shift(1); prior_ret=p.pct_change(20).shift(1)
recovery=(p.shift(1)/prior_min-1)/(vol*np.sqrt(20)+1e-12); f=recovery.where(prior_ret<0).rolling(3,min_periods=3).mean().shift(1); fr=p.shift(-10)/p-1
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],fr.loc[d])
 if pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr.loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[ic(f.loc[d],y.loc[d]) for d in x.index]; q=[v for v in q if pd.notna(v)]; print('decay',h,np.mean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2030',z[z.index.year==2030]),('2031',z[z.index.year==2031]),('2032',z[z.index.year==2032])]: print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_3_20320513_drawdown_recovery_signal.csv'); x.to_csv('scripts/miner_3_20320513_drawdown_recovery_ic.csv')
