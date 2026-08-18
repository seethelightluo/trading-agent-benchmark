import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 p=('../persistent/index_data/' if s=='DXY' else '../persistent/stock_data/')+s+'.csv'
 if not os.path.exists(p):return pd.Series(dtype=float)
 d=pd.read_csv(p);return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.DataFrame({s:get(s) for s in U});dxy=get('DXY');px=px.loc[:'2026-07-15'];dxy=dxy.loc[:'2026-07-15'];r=px.pct_change();dr=dxy.pct_change();rows=[];out=[]
for dt in px.index:
 if dt not in dr.index:continue
 h=r.loc[:dt].tail(60);dh=dr.loc[:dt].tail(60)
 if len(h)<50 or len(dh)<50:continue
 d20=px.loc[:dt].pct_change(20).iloc[-1].values;x=h.fillna(0).values;y=dh.reindex(h.index).fillna(0).values;den=((y-y.mean())**2).sum();b=((x-x.mean(0))*(y-y.mean())[:,None]).sum(0)/den if den>1e-12 else np.zeros(15);sig=d20-b*dxy.loc[:dt].pct_change(20).iloc[-1]
 out += [{'date':dt,'symbol':s,'value':v} for s,v in zip(U,sig)]
 z=pd.DataFrame({'s':sig,'y':r.shift(-1).loc[dt].values}).dropna()
 if len(z)>=8:rows.append((dt,z.s.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(a),'mean_n',a.n.mean(),'mean_ic',a.ic.mean(),'icir',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for y,g in a.groupby(a.index.year):print(y,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),4))
pd.DataFrame(out).to_csv('scripts/miner_1_20260730_dollar_residual_momentum_signals.csv',index=False);print('artifact rows',len(out))
