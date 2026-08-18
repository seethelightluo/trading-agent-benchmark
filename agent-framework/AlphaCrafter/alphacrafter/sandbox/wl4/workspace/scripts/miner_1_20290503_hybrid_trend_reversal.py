import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d['date'])
 d=d[d.date<=pd.Timestamp('2029-05-02')].sort_values('date').set_index('date'); px[a]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); trend=P.pct_change(20); short=P.pct_change(3); vol=R.rolling(20).std()
F=((trend-short)/(np.sqrt(20)*vol)).clip(-8,8)
def calc(h):
 out=[]
 for i in range(len(P)-h):
  v=F.iloc[i]; fw=P.iloc[i+h]/P.iloc[i]-1; ok=v.notna()&fw.notna()
  if ok.sum()>=8: out.append(spearmanr(v[ok],fw[ok]).statistic)
 z=np.array(out); return z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),len(z)
for h in [1,5,10,20]: print('horizon',h,'IC ICIR hit obs',calc(h))
rows=[]
for i in range(len(P)-10):
 v=F.iloc[i]; fw=P.iloc[i+10]/P.iloc[i]-1; ok=v.notna()&fw.notna()
 if ok.sum()>=8: rows.append((P.index[i],spearmanr(v[ok],fw[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rank=F.rank(axis=1,pct=True)
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.sum()/(len(x)*15),'turnover_proxy',rank.diff().abs().mean(axis=1).dropna().mean())
print('regimes',x.iloc[:len(x)//2].ic.mean(),x.iloc[len(x)//2:].ic.mean(),'recent250',x.tail(250).ic.mean(),'period',x.index.min().date(),x.index.max().date())
