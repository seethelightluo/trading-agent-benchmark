import pandas as pd
p='scripts/miner_2_20330527_vix_conditioned_reversal.py'
s=open(p).read(); s=s.replace("print('dates',len(dates),'rows',len(df),'active_dates',df.date.nunique(),'avg_n',df.groupby('date').size().mean())", "df[df.h==5][['date','s','sig']].to_csv('scripts/miner_2_20330527_vix_conditioned_reversal_signal.csv',index=False)\nprint('dates',len(dates),'rows',len(df),'active_dates',df.date.nunique(),'avg_n',df.groupby('date').size().mean())")
open(p,'w').write(s)
PY
python scripts/miner_2_20330527_vix_conditioned_reversal.py
ls scripts/miner_2_20330527_vix_conditioned_reversal_signal.csv