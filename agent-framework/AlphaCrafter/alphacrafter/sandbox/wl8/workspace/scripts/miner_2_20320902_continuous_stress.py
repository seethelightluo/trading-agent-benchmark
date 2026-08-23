import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-09-01')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
r=p.pct_change(); vl=v.shift(1)
# Continuous stress intensity: lagged 120d VIX percentile, centered at 0.5, scales contrarian 5d return; 3d smoothing.
rank=vl.rolling(120,min_periods=60).apply(lambda x: (x<=x[-1]).mean(),raw=True)
f=(-r.shift(1).rolling(5,min_periods=5).sum()).mul((rank-0.5),axis=0).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]}; rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-06-01') or d>cut or i+10>=len(p):continue
 q=ic(f.loc[d],fr[10].loc[d])
 if pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr[10].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x["ic"]
m=float(z.mean()); ir=float(m/z.std(ddof=1)); turn=float(f.rank(pct=True).diff().abs().mean().mean());cov=float(f.loc[x.index].notna().mean().mean())
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'IC',m,'ICIR',ir,'turnover',turn,'coverage',cov)
for h in [1,5,10,20]:
 q=[ic(f.loc[d],fr[h].loc[d]) for d in x.index];q=[a for a in q if pd.notna(a)];print('decay',h,float(np.mean(q)))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,float(q.mean()),float(q.mean()/q.std(ddof=1)),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20320902_continuous_stress_signal.csv');x.to_csv('scripts/miner_2_20320902_continuous_stress_ic.csv')
print('METRICS',json.dumps({'ic':m,'icir':ir,'turnover':turn,'coverage':cov,'dates':len(z),'avg_n':0.0,'cutoff':str(cut.date())}))
