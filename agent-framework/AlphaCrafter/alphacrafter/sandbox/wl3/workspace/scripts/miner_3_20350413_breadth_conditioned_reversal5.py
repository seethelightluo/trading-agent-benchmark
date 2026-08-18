import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-04-13')]; r=c.pct_change(); v=r.rolling(20,min_periods=15).std()
# Cross-asset panic conditioning: fade 5-session moves more aggressively when downside breadth is unusually wide.
b=(r<0).mean(axis=1); mu=b.rolling(120,min_periods=60).mean(); sd=b.rolling(120,min_periods=60).std(); z=((b-mu)/(sd+1e-12)).clip(-2,2)
s=(-r.rolling(5,min_periods=5).sum()/(v*np.sqrt(20)+1e-12)).mul((1+.40*z).clip(.45,1.55),axis=0)
f=c.pct_change(5).shift(-5); out=[]
for dt in s.index:
 ok=s.loc[dt].notna()&f.loc[dt].notna()
 if ok.sum()>=8: out.append((dt,spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic,ok.sum()))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=r.ic.dropna(); print('factor=breadth_conditioned_volscaled_reversal5d'); print('dates',len(r),'instruments',15,'avg_n',r.n.mean(),'coverage',r.n.mean()/15); print('daily_ic %.8f daily_icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 f=c.pct_change(h).shift(-h); z2=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:z2.append(spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic)
 print('horizon',h,'ic',np.nanmean(z2),'n',len(z2))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),x.mean(),x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),(x>0).mean())
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',r.index.min().date(),r.index.max().date())
