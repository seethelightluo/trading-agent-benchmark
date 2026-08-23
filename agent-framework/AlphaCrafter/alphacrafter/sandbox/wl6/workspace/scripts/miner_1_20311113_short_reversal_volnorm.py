import os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    p=os.path.join(base,s+'.csv')
    if not os.path.exists(p):
        p=os.path.join(base,s.replace('/','_')+'.csv')
    if os.path.exists(p):
        d=pd.read_csv(p)
        datecol='date' if 'date' in d else d.columns[0]
        closecol='close' if 'close' in d else 'Close'
        d['date']=pd.to_datetime(d[datecol]); d['close']=pd.to_numeric(d[closecol],errors='coerce')
        px[s]=d.set_index('date')['close'].sort_index()
P=pd.DataFrame(px).sort_index().ffill()
P=P.loc[:'2031-10-29']
r=P.pct_change()
# candidate: short reversal, damped by volatility (higher = expected forward return)
factor=-(P/P.shift(5)-1)/(r.rolling(20,min_periods=10).std()+1e-12)
rows=[]
for h in [5,10,20]:
    ics=[]; cov=[]; turns=[]
    for i in range(20,len(P)-h):
        x=factor.iloc[i]; y=(P.iloc[i+h]/P.iloc[i]-1)
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
            cov.append(len(z)/15)
            if i>20:
                prev=factor.iloc[i-1].reindex(z.index)
                turns.append(np.mean(np.sign(z.iloc[:,0])!=np.sign(prev)))
    a=np.array(ics); print({'horizon':h,'dates':len(a),'avg_instruments':round(np.mean(cov)*15,2),'coverage':round(np.mean(cov),4),'ic':round(np.nanmean(a),8),'icir':round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(len(a)),4),'hit':round(np.mean(a>0),4),'turnover':round(np.mean(turns),4)})
# annual IC for selected 10
h=10
for yr in sorted(set(P.index.year)):
    a=[]
    for i in range(20,len(P)-h):
        if P.index[i].year!=yr: continue
        z=pd.concat([factor.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
        if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    if a: print('year',yr,'n',len(a),'ic',round(float(np.mean(a)),6))
