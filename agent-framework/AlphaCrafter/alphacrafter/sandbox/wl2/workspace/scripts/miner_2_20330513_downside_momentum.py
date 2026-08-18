import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-05-13')
pdct={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}; p=pd.DataFrame(pdct).sort_index();p=p[p.index<=cut];r=p.pct_change()
# Trend continuation scored by return over 20d divided by downside volatility over 40d;
# lagged cross-sectional ranks avoid scale effects and use only observable history.
down=r.clip(upper=0).rolling(40).std(); f=p.pct_change(20)/down
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,3,5,10]:
 y=p.pct_change(h).shift(-h); ii=[]; nn=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:ii.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
 x=np.array(ii);print(h,'dates',len(x),'avg_n',round(np.mean(nn),2),'coverage',round(np.mean(nn)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for s,e in [('2026-01-01','2029-12-31'),('2030-01-01','2033-05-13')]:
 x=[]
 for d in f.index:
  if pd.Timestamp(s)<=d<=pd.Timestamp(e):
   z=pd.concat([f.loc[d],r.shift(-1).loc[d]],axis=1).dropna()
   if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('regime',s,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
