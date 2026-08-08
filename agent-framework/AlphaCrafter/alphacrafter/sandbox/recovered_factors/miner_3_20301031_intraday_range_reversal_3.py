import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
# Intraday signed reversal pressure: negative of recent close-open move normalized by true range, lagged via using t and forward t+1
rows=[]; horizon=5
for a,df in data.items():
    op,cl,hi,lo=df.open,df.close,df.high,df.low
    # robust intraday return / range, smoothed 3 completed sessions
    pressure=((cl-op)/(hi-lo).replace(0,np.nan)).rolling(3,min_periods=2).mean()
    fac=-pressure
    fr=cl.shift(-horizon)/cl-1
    z=pd.DataFrame({'f':fac,'r':fr,'a':a}).dropna(); rows.append(z)
x=pd.concat(rows)
ics=[]; counts=[]
for d,g in x.groupby(level=0):
    if len(g)>=8:
        ics.append(spearmanr(g.f,g.r).statistic); counts.append(len(g))
ics=np.array(ics)
print('candidate=intraday_range_reversal_3; dates',len(ics),'meanN',np.mean(counts),'cells',len(x),'coverage',len(x)/(len(set(x.index))*15))
print('H5 IC %.6f ICIR %.6f hit %.4f'%(np.mean(ics),np.mean(ics)/np.std(ics,ddof=1),np.mean(ics>0)))
for h in [1,5,10,20]:
 rr=[]
 for a,df in data.items():
  pressure=((df.close-df.open)/(df.high-df.low).replace(0,np.nan)).rolling(3,min_periods=2).mean()
  z=pd.DataFrame({'f':-pressure,'r':df.close.shift(-h)/df.close-1}).dropna(); z['date']=z.index; rr.append(z)
 q=pd.concat(rr); ii=[]
 for d,g in q.groupby('date'):
  if len(g)>=8: ii.append(spearmanr(g.f,g.r).statistic)
 ii=np.array(ii); print('h',h,'dates',len(ii),'IC',round(np.mean(ii),6),'ICIR',round(np.mean(ii)/np.std(ii,ddof=1),6),'hit',round(np.mean(ii>0),4))
# regimes H5
for name,mask in [('2020-23',x.index<'2024-01-01'),('2024-27',(x.index>='2024-01-01')&(x.index<'2028-01-01')),('2028+',x.index>='2028-01-01'),('latest120',x.index>=x.index.unique().sort_values()[-120])]:
 ii=[]
 for d,g in x[mask].groupby(level=0):
  if len(g)>=8: ii.append(spearmanr(g.f,g.r).statistic)
 ii=np.array(ii); print(name,len(ii), 'IC',round(np.mean(ii),6) if len(ii) else None,'ICIR',round(np.mean(ii)/np.std(ii,ddof=1),6) if len(ii)>1 else None)
# turnover cross section rank proxy
wide=x.reset_index().pivot(index='date',columns='a',values='f'); ranks=wide.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean())
# correlation to rough existing common signals: factor versus each factor's own generic? pooled candidate vs close momentum 20, volnorm reversal 5, trend consistency
cand=[]; others={k:[] for k in ['mom20','rev5','trend20','vol40']}
for a,df in data.items():
 f=-((df.close-df.open)/(df.high-df.low).replace(0,np.nan)).rolling(3,min_periods=2).mean()
 cand.append(f.rename(a))
 others['mom20'].append(df.close.pct_change(20).rename(a)); others['rev5'].append((-df.close.pct_change(5)).rename(a)); others['trend20'].append((df.close.pct_change(20)/df.close.pct_change().rolling(20).std()).rename(a)); others['vol40'].append((-df.close.pct_change().rolling(40).std()).rename(a))
cc=pd.concat(cand,axis=1); print('rough_library_max_abs',max(abs(spearmanr(cc.stack(),pd.concat(v,axis=1).stack()).statistic) for v in others.values()))
