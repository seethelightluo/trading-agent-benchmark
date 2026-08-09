# append sensitivity of macro-conditioned reversal
exec(open('scripts/miner_2_20260716_macro_impulse.py').read().split("for h in [1,5,10]:\n a=[]\n for i in range(len(px)-h):\n  q=pd.concat([fac2")[0])
for c in [0.5,1.0,2.0]:
 f=rev*(1+c*shock.abs().values[:,None]*(-bs)); a=[]
 for i in range(len(px)-1):
  q=pd.concat([f.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a); print('conditional_rev_c',c,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
