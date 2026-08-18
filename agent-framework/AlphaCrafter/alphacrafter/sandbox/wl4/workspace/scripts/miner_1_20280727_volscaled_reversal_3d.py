import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];b='../persistent/stock_data'
P=pd.DataFrame({a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).sort_index().loc[:'2028-07-26'];r=P.pct_change();
# Three-day reversal, divided by 20-day volatility; cross-sectional z-score and one-day lag.
sig=-(P.pct_change(3))/(r.rolling(20,min_periods=15).std()+1e-12);sig=sig.sub(sig.mean(1),axis=0).div(sig.std(1)+1e-12,axis=0).shift(1)
def ic(x,y):
 o=np.isfinite(x)&np.isfinite(y)
 return (np.corrcoef(pd.Series(x[o]).rank(),pd.Series(y[o]).rank())[0,1],o.sum()) if o.sum()>=8 else (np.nan,o.sum())
print('rows',len(P),'assets',len(A),'end',P.index.max().date())
for h in [1,5,10,20]:
 v=[];n=[]
 for i in range(len(P)-h):
  q,k=ic(sig.iloc[i].values,(P.iloc[i+h]/P.iloc[i]-1).values)
  if np.isfinite(q):v.append(q);n.append(k)
 s=pd.Series(v);z=s.tail(250);print('h',h,'dates',len(s),'avg_n',round(np.mean(n),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',round(sig.notna().sum().sum()/(len(sig)*len(A)),4),'turnover',round(sig.rank(1,pct=True).diff().abs().mean(1).dropna().mean(),6))
