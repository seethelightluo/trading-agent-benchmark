import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2031-11-27')
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1).sort_index().loc[:E]
R=np.log(P).diff(); V=R.rolling(20).std().shift(1); M=np.log(P/P.shift(20)).shift(1)/V
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.sort_index().loc[:E]; s=v.pct_change().abs().rolling(252,min_periods=126).apply(lambda z:(z[:-1]<=z[-1]).mean(),raw=True).shift(1)
F=M.mul(1+0.75*s.reindex(P.index).ffill(),axis=0); out=[]; nn=[]
for d in F.index:
 z=pd.concat([F.loc[d],np.log(P.shift(-10)/P).loc[d]],axis=1).dropna()
 if len(z)>=8:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
a=np.array(out);print({'dates':len(a),'avg_names':round(np.mean(nn),2),'coverage':round(F.notna().mean().mean(),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4)})
for n in [180,365,756]:
 b=a[-n:];print(n,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),len(b))
