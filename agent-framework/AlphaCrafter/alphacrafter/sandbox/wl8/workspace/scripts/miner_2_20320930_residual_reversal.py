import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-09-20')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change()
# Residual reversal: remove contemporaneous cross-asset common return, then fade 10d residual moves.
common=r.mean(axis=1)
res=r.sub(common,axis=0)
vol=r.rolling(20,min_periods=20).std()
f=(-res.shift(1).rolling(10,min_periods=10).sum()).div(vol.shift(1).replace(0,np.nan),axis=0)
# mild smoothing reduces one-day rank noise
f=f.rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for d in p.index[:-20]:
 if d<pd.Timestamp('2020-06-01') or d>cut:continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q): rows.append((d,q,(f.loc[d].notna()&(p.shift(-10)/p-1).loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
m=float(z.mean()); ir=float(m/z.std(ddof=1)); turn=float(f.rank(pct=True).diff().abs().mean().mean()); cov=float(f.loc[x.index].notna().mean().mean())
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',float(x.n.mean()),'IC',m,'ICIR',ir,'turnover',turn,'coverage',cov)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; q=[ic(f.loc[d],fr.loc[d]) for d in x.index];q=[a for a in q if pd.notna(a)];print('decay',h,float(np.mean(q)))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,float(q.mean()),float(q.mean()/q.std(ddof=1)),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20320930_residual_reversal_signal.csv');x.to_csv('scripts/miner_2_20320930_residual_reversal_ic.csv')
print('METRICS',json.dumps({'ic':m,'icir':ir,'turnover':turn,'coverage':cov,'dates':len(z),'avg_n':float(x.n.mean()),'cutoff':str(cut.date())}))
