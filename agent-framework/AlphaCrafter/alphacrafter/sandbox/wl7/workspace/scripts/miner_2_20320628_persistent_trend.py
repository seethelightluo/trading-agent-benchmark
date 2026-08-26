import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut='2032-06-27'
frames={}
for p in glob.glob('../persistent/stock_data/*.csv'):
    s=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
    d=d.loc[:cut]; frames[s]=d['close']
px=pd.DataFrame(frames).sort_index()
# Trend persistence: medium-term return weighted by fraction of positive daily moves,
# normalized by lagged 30d volatility. All inputs available at signal date.
r=px.pct_change()
ret20=px/px.shift(20)-1
posfrac=r.gt(0).rolling(20,min_periods=15).mean()
vol=r.rolling(30,min_periods=20).std()
sig=(ret20*(0.5+posfrac)/vol).replace([np.inf,-np.inf],np.nan)
print('cutoff',cut,'assets',len(px.columns),'rows',len(px))
print('signal dates',int(sig.notna().sum().sum()),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
    fwd=px.shift(-h)/px-1
    vals=[]; ns=[]; turns=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8:
            vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
            turns.append((x[ok].rank().ne(x[ok].rank().shift()).mean()))
    a=np.array(vals); ic=np.nanmean(a); ir=ic/np.nanstd(a,ddof=1)*np.sqrt(252) if len(a)>1 else np.nan
    print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,np.mean(a>0)))
    # thirds
    z=np.array_split(a,3); print(' thirds',[(round(np.nanmean(q),6),len(q)) for q in z])
# signal artifact for admission horizon 10
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320628_persistent_trend_signal.csv',index=False)
