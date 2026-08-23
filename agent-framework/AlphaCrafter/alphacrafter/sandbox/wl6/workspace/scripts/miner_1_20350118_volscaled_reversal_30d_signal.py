import pandas as pd
p=pd.read_csv('scripts/miner_1_20350118_volscaled_momentum_30d_signal.csv')
p['signal']=-p['signal']
p.to_csv('scripts/miner_1_20350118_volscaled_reversal_30d_signal.csv',index=False)
print('rows',len(p),'dates',p.date.nunique(),'symbols',p.symbol.nunique())
