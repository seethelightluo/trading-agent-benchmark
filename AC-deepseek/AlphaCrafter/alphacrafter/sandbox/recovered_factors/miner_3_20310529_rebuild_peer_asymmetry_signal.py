import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def c(a):
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return x.close.astype(float)
C=pd.DataFrame({a:c(a) for a in A});r=np.log(C).diff(); med=r.median(axis=1);F=pd.DataFrame(index=C.index,columns=A,dtype=float)
for a in A:
 peer=r.drop(columns=a).mean(axis=1); vals=[]
 for i in range(len(r)):
  z=pd.DataFrame({'x':r[a].iloc[max(0,i-59):i+1],'p':peer.iloc[max(0,i-59):i+1],'m':med.iloc[max(0,i-59):i+1]}).dropna();dn=z[z.m<0];up=z[z.m>0]
  vals.append(dn.x.corr(dn.p)-up.x.corr(up.p) if len(dn)>=12 and len(up)>=12 else np.nan)
 F[a]=vals
F.to_pickle('scripts/miner_2_20310515_inverse_peer_up_down_comovement_asymmetry_60obs_signal.pkl')
print('saved',F.shape,float(F.notna().mean().mean()))
