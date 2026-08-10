import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(p)[:-4] for p in files]
C={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Low-dispersion trend continuation: 10d momentum, volatility normalized,
# activated when cross-asset 5d dispersion is below its trailing 60d median.
mom=close.pct_change(10); vol=r.rolling(20).std(); raw=mom/vol.replace(0,np.nan)
disp=r.rolling(5).std().mean(axis=1)
threshold=disp.rolling(60,min_periods=30).median()
gate=(disp<=threshold).astype(float)
fac=raw.mul(gate,axis=0)
fac.to_csv('scripts/miner_2_20270325_lowdisp_trend_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); ds.append(dt)
 s=pd.Series(vals,index=ds); return s,ns
print('assets',len(assets),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean(),'active_dates',int((gate>0).sum()))
