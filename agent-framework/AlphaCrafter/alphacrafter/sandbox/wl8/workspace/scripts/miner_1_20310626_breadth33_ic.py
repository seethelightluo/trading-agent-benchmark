import pandas as pd
exec(open('scripts/miner_1_20310626_breadth_thresholds.py').read().split("for th in [.33,.40,.50]:")[0])
rows=[]
for dt in P.index:
 a=mom.loc[dt] if br.loc[dt]>=.33 else -mom.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok],method='spearman'),int(ok.sum())))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20310626_breadth33_momentum_ic.csv',index=False)
