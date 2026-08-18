import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): return None
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].sort_index()
px=pd.concat({s:load(s) for s in U},axis=1).dropna(how='all')
dxy=pd.read_csv('../persistent/index_data/DXY.csv'); dxy['date']=pd.to_datetime(dxy['date']); dxy=dxy.set_index('date')['close'].sort_index()
r=np.log(px).diff(); mr=np.log(dxy).diff().reindex(r.index)
# rolling beta and residual 5d move; signal reversal of macro residual shock
bet=r.rolling(60,min_periods=40).cov(mr).div(mr.rolling(60,min_periods=40).var(),axis=0)
res=(r.rolling(5).sum()-bet*r.groupby(level=0).rolling(5).sum().sum() if False else None)
# aligned DXY cumulative 5d
m5=mr.rolling(5).sum(); a5=r.rolling(5).sum(); resid=a5.sub(bet.mul(m5,axis=0))
factor=-resid.shift(1)
rows=[]
for dt in r.index:
 vals=factor.loc[dt]; fwd=r.shift(-1).loc[dt]
 ok=vals.notna()&fwd.notna()
 if ok.sum()>=8:
  rows.append((dt,spearmanr(vals[ok],fwd[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/len(U),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'recent',x.tail(500).ic.mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-04-01')]:
 z=x.loc[a:b]; print(a,'n',len(z),'ic',z.ic.mean(),'ir',z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
print('decay')
for h in [1,3,5,10]:
 fw=r.rolling(h).sum().shift(-h)
 q=[]
 for dt in r.index:
  v=factor.loc[dt]; y=fw.loc[dt]; ok=v.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],y[ok]).statistic)
 print(h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
# artifact signal
factor.to_csv('scripts/miner_3_20330401_macro_dxy_residual_reversal_signal.csv')
