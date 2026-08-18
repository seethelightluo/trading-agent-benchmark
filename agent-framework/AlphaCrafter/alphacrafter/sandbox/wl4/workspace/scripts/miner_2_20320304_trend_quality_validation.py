import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    p=f'../persistent/stock_data/{s}.csv'
    if os.path.exists(p):
        x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.concat(D,axis=1).sort_index().loc[:'2032-03-03']
# medium trend quality: lagged 60d log return divided by lagged 20d realized vol
lr=np.log(px).diff(); ret60=np.log(px/px.shift(60)); vol20=lr.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(ret60/vol20).shift(1)
# evaluate H 5,10,20, using only dates with >=8 names
for h in [5,10,20]:
    fr=np.log(px.shift(-h)/px)
    ics=[]; ns=[]; turns=[]
    prev=None
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
        # rank turnover across valid names
        r=f.loc[dt].rank(pct=True)
        if prev is not None:
            q=pd.concat([r,prev],axis=1).dropna(); turns.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
        prev=r
    a=np.array(ics); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/(np.nanstd(a,ddof=1)/np.sqrt(len(a))):.4f} hit={np.mean(a>0):.4f} turnover={np.nanmean(turns):.4f}')
    if h==10:
        for n in [260,520,780]:
            x=a[-n:]; print(' recent',n,'IC',np.mean(x),'ICIR',np.mean(x)/(np.std(x,ddof=1)/np.sqrt(len(x))),'hit',np.mean(x>0))
print('cutoff',px.index.max().date(),'assets',px.shape[1])
