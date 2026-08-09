import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); p[s]=d.set_index('date')['close'].sort_index()
p=pd.DataFrame(p).sort_index().ffill().loc[:'2026-12-17'];v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill();vp=v.rolling(126,min_periods=63).rank(pct=True);f=(-p.pct_change(3)).mul((.5+vp).clip(.5,1.5),axis=0).shift(1)
for h in [1,5,10]:
 fr=p.shift(-h)/p-1;z=[];n=[]
 for dt in f.index:
  a=f.loc[dt];b=fr.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(a[ok].rank().corr(b[ok].rank()));n.append(ok.sum())
 x=np.array(z);print(h,len(x),np.mean(n),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12),np.mean(x>0))
print('coverage',f.notna().sum(axis=1).mean()/15)
