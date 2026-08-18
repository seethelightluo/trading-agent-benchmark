import pandas as pd
p='../persistent/miner_2_20350427_stress_relative_strength_signal.csv'
f=pd.read_csv(p,index_col=0)
f=-f
f.to_csv('../persistent/miner_2_20350427_stress_relative_weakness_reversal_signal.csv',index_label='date')
print('saved',f.shape,'coverage',f.notna().mean().mean())
