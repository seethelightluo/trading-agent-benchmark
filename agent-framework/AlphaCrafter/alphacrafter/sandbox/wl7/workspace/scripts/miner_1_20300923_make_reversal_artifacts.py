import pandas as pd
s=pd.read_csv('scripts/miner_1_20300923_vol_scaled_trend20_40_signal.csv',index_col=0)
s=-s
s.to_csv('scripts/miner_1_20300923_vol_scaled_reversal20_40_signal.csv')
i=pd.read_csv('scripts/miner_1_20300923_vol_scaled_trend20_40_ic.csv');i['ic']=-i['ic'];i.to_csv('scripts/miner_1_20300923_vol_scaled_reversal20_40_ic.csv',index=False)
print('wrote',s.shape,len(i),'signal rows and IC rows')
