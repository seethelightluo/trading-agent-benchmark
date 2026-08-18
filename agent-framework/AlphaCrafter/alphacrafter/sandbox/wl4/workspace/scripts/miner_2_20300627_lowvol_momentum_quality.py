import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data/'
px=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2030-06-26']; r=px.pct_change()
# persistent quality trend: medium trend divided by realized risk, with a mild short-term anti-chase term
f=(px.pct_change(60)/(r.rolling(60).std()*np.sqrt(60)))-0.25*px.pct_change(5)/(r.rolling(20).std()*np.sqrt(5))
for h in [10,20]:
 out=[]
 for i in range(70,len(px)-h):
  x=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.array(out); print('horizon',h,'dates',len(a),'avgN', '15 approx','IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean(),'recent252',a[-252:].mean(),a[-252:].mean()/a[-252:].std(ddof=1)*np.sqrt(min(252,len(a))))
print('decay')
for h in [1,5,10,20]:
 out=[]
 for i in range(70,len(px)-h):
  x=f.iloc[i];y=px.iloc[i+h]/px.iloc[i]-1;ok=x.notna()&y.notna()
  if ok.sum()>=8:out.append(spearmanr(x[ok],y[ok]).statistic)
 print(h,np.nanmean(out))
