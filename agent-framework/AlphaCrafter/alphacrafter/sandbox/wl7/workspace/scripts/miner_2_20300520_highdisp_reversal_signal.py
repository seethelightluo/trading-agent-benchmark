import pandas as pd, numpy as np
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); lr=np.log(P).diff(); r5=lr.rolling(5).sum(); vol=lr.rolling(20).std()
disp=lr.rolling(20).std().mean(axis=1); cutoff=disp.rolling(252,min_periods=60).quantile(.8)
f=(-r5/(vol+1e-12)).mul((disp>cutoff).astype(float),axis=0).shift(1)
out=f.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_2_20300520_highdisp_reversal_signal.csv',index=False)
print('signal_rows',len(out),'valid_fraction',f.notna().mean().mean())
