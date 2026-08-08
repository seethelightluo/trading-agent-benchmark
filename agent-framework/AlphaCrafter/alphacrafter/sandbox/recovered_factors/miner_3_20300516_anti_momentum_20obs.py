import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob, os
files=glob.glob('../persistent/stock_data/*.csv')
px={}
for f in files:
    s=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
    px[os.path.basename(f)[:-4]]=s
p=pd.DataFrame(px).sort_index()
# explicit anti-momentum: negative trailing 20-observation return, computed at t
sig=-(p/p.shift(20)-1)
rets=p.pct_change()
print('assets',p.shape[1],'dates',p.index.min().date(),p.index.max().date())
for h in [1,3,5,10,20]:
    # forward compounded return strictly after t
    fwd=p.shift(-h)/p-1
    ics=[]; ns=[]; turnovers=[]
    prev=None
    for d in p.index:
        x=sig.loc[d]; y=fwd.loc[d]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
        cs=x.rank(pct=True)
        if prev is not None:
            q=pd.concat([prev,cs],axis=1).dropna(); turnovers.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
        prev=cs
    a=np.array(ics); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),3),'turn',round(np.nanmean(turnovers),4))
    for name,mask in [('2024-2027',(p.index>='2024-01-01')&(p.index<='2027-12-31')),('2028-2030',p.index>='2028-01-01'),('latest120',np.arange(len(p))>=len(p)-120)]:
      ds=p.index[mask]; vals=[]
      for d in ds:
       z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
       if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
      aa=np.array(vals); print(' ',name,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else np.nan)
# asset class diagnostics H10
h=10; fwd=p.shift(-h)/p-1
for group in [['BTC','ETH'],['XAU','COPPER','WTI'],['SPX','NDX','SOX','000300.SH','000688.SH','HSI','N225','SX5E'],['US10Y','CN10Y']]:
 vals=[]
 for d in p.index:
  z=pd.concat([sig.loc[d,group],fwd.loc[d,group]],axis=1).dropna()
  if len(z)>=2: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('group',group,'n',len(vals),'IC',np.nanmean(vals))
