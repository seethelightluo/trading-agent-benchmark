import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-31')
O={}; C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 O[s]=d.open; C[s]=d.close
op=pd.concat(O,axis=1).reindex(columns=U); cl=pd.concat(C,axis=1).reindex(columns=U)
intr=cl/op-1; common=intr.mean(axis=1); res=intr.sub(common,axis=0)
# inverse volatility scales idiosyncratic intraday reversal, with 20d completed-session volatility
iv=1/res.rolling(20,min_periods=15).std().clip(upper=100)
f=(-res*iv).replace([np.inf,-np.inf],np.nan)
y=cl.pct_change(1).shift(-1)
vals=[]; dates=[]; ns=[]
for i in range(len(cl)-1):
 q=pd.concat([f.iloc[i],y.iloc[i].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:
  r=spearmanr(q.iloc[:,0],q.y).statistic
  if np.isfinite(r): vals.append(r); dates.append(cl.index[i]); ns.append(len(q))
a=np.array(vals); dt=np.array(dates)
print('candidate=vol_scaled_residual_intraday_reversal; dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 aa=a[(dt>=pd.Timestamp(lo))&(dt<=pd.Timestamp(hi))]
 print('regime',lo,'n',len(aa),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6))
ranks=f.rank(pct=True); print('coverage',round(f.notna().mean().mean(),5),'turnover',round(np.nanmean(np.abs(ranks.diff()).mean(axis=1)),5))
for h in [3,5]:
 yy=cl.pct_change(h).shift(-h); z=[]
 for i in range(len(cl)-h):
  q=pd.concat([f.iloc[i],yy.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: z.append(spearmanr(q.iloc[:,0],q.y).statistic)
 z=np.array(z); print('horizon',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
