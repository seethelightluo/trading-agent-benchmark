import pandas as pd
p='scripts/miner_1_20291008_recovery_adjusted_trend_ic.csv'
x=pd.read_csv(p); x['ic']=-x['ic']; x.to_csv('scripts/miner_1_20291008_recovery_adjusted_reversal_ic.csv',index=False)
print('reversal artifact rows',len(x),'mean_ic',round(x.ic.mean(),6),'icir',round(x.ic.mean()/x.ic.std(ddof=1),6))
