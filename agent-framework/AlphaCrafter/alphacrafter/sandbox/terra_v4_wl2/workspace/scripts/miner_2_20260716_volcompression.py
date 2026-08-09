import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in syms}; px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# volatility compression: lower recent vol than longer baseline, lagged only
vol20=r.rolling(20).std(); vol60=r.rolling(60).std(); f=-(vol20/vol60-1)
# volatility-adjusted carry: 20d return / vol, likely existing momentum but test 5d forward
for nm,x in [('compression',f),('lowvol',-vol20),('volshock',-(vol5:=r.rolling(5).std())/vol20)]:
 print('\n'+nm)
 for h in [1,5,10]:
  y=px.pct_change(h).shift(-h); z=[]; ns=[]
  for d in x.index:
   ok=x.loc[d].notna()&y.loc[d].notna()
   if ok.sum()>=8:z.append(spearmanr(x.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
  z=pd.Series(z).dropna(); print(h,len(z),round(z.mean(),5),round(z.mean()/z.std(ddof=1),5),round((z>0).mean(),4),round(np.mean(ns),2))
 print('turn',x.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',x.notna().sum(axis=1).mean()/15)
