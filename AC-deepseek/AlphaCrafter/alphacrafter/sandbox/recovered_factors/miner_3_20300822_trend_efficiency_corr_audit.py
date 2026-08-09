import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float); r=p.pct_change()
ret20=p.shift(1)/p.shift(21)-1; path=r.abs().rolling(20,min_periods=15).sum().shift(1); sig=ret20/path
vol=r.rolling(20,min_periods=15).std().shift(1)
lib={'risk_adjusted_trend_20d':ret20/vol,'ravmom_20obs':ret20/vol,'ret20':ret20}
for name,c in lib.items():
 z=pd.concat([sig.stack().rename('candidate'),c.stack().rename('library')],axis=1).dropna(); print(name,round(abs(spearmanr(z.candidate,z.library).statistic),6),len(z))
