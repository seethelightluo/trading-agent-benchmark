import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
universe=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in universe:
    d=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: d=fn(s, days=3000)
        except Exception: pass
        if d is not None and len(d): break
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(frames).sort_index(); ret=px.pct_change(); csdisp=ret.rolling(20,min_periods=15).std().mean(axis=1); threshold=csdisp.rolling(120,min_periods=60).median(); gate=(csdisp>threshold).astype(float)
factor=-(px/px.shift(5)-1)/ret.rolling(20,min_periods=15).std(); factor=factor.mul(gate,axis=0)
for h in [5,10,20]:
    ics=[]; ninst=[]
    for i in range(len(px)-h):
        z=pd.concat([factor.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ninst.append(len(z))
    a=np.array(ics); mean=np.nanmean(a); sd=np.nanstd(a,ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    ranks=factor.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean(); cov=factor.notna().sum().sum()/(factor.shape[0]*len(universe))
    print(f'h={h} dates={len(a)} avg_inst={np.mean(ninst):.3f} coverage={cov:.4%} IC={mean:.8f} ICIR={icir:.8f} hit={np.mean(a>0):.4%} turnover={turnover:.8f}')
for yr in sorted(set(px.index.year)):
    ics=[]
    for i in range(len(px)-20):
        if px.index[i].year!=yr: continue
        z=pd.concat([factor.iloc[i],px.iloc[i+20]/px.iloc[i]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    if ics: print('regime',yr,'n',len(ics),'ic',np.nanmean(ics))
print('assets',len(frames),'rows',len(px),'through',px.index.max().date())
