import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2027-09-22')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
C=pd.concat({s:load(s).close for s in A},axis=1).loc[:end].sort_index(); R=C.pct_change()
# Directional persistence: lagged net 30d return weighted by path efficiency, damped by realized risk.
net=C.pct_change(30); path=R.abs().rolling(30).sum(); efficiency=net.abs()/(path+1e-9)
vol=R.rolling(30).std(); f=(net*efficiency)/(vol+0.005); f=f.shift(1)
rows=[]
for s in A:
 for h in [5,10,20]:
  fw=C[s].shift(-h)/C[s]-1
  for dt in f.index:
   if pd.notna(f.loc[dt,s]) and pd.notna(fw.loc[dt]): rows.append((dt,s,float(f.loc[dt,s]),h,float(fw.loc[dt])))
df=pd.DataFrame(rows,columns=['date','symbol','factor','h','fwd'])
for h,g in df.groupby('h'):
 vals=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8:
   z=spearmanr(x.factor,x.fwd).statistic
   if np.isfinite(z): vals.append(z)
 a=np.array(vals); print('horizon',h,'dates',len(a),'avg_n',g.date.nunique(),'avg_inst',len(g)/max(1,len(a)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
p=df[df.h==20].pivot(index='date',columns='symbol',values='factor'); ranks=p.rank(axis=1,pct=True)
print('cutoff',end.date(),'dates',df.date.nunique(),'symbols',df.symbol.nunique(),'coverage',p.notna().mean().mean(),'turnover',ranks.diff().abs().mean().mean())
df[df.h==20].to_csv('scripts/miner_2_20270923_path_efficiency_signal.csv',index=False)
for label,mask in [('2026+',df.date>='2026-01-01'),('2027',df.date>='2027-01-01'),('Q2+',df.date>='2027-04-01')]:
 g=df[(df.h==20)&mask]; vals=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8: vals.append(spearmanr(x.factor,x.fwd).statistic)
 a=np.array(vals);print(label,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
