import pandas as pd
p='scripts/miner_3_20310519_trend_acceleration_signal.csv'
d=pd.read_csv(p,index_col=0)
(-d).to_csv('scripts/miner_3_20310519_trend_acceleration_reversal_signal.csv')
print('wrote reversed signal',d.shape)
