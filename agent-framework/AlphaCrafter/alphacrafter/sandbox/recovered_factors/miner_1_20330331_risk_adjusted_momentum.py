import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
K=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in K:
  x=pd.read_csv(f);x.date=pd.to_datetime(x.date);d[s]=x.set_index('date').close
p=pd.DataFrame(d).sort_index().loc[:'2033-03-30'];r=p.pct_change();
# Risk-adjusted momentum: medium return divided by realized volatility, with volatility floor, lagged.
f=(r.rolling(10,min_periods=8).sum()/r.rolling(20,min_periods=15).std().clip(lower=.003)).shift(1)
print('candidate risk_adjusted_momentum_10_20; dates',len(p),'assets',len(p.columns))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;a=[];n=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);n.append(len(q))
 a=np.array(a);print('H',h,'dates',len(a),'meanN',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
