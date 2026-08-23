
# Persist signal provenance artifacts for deterministic audit.
rows=[]; q=np.log(p.shift(-10)/p)
for d in f.index:
 z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8:
  for s in z.index: rows.append((d,s,float(f.loc[d,s])))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20330303_simple_downside_transition_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values}).to_csv('scripts/miner_2_20330303_simple_downside_transition_ic.csv',index=False)
