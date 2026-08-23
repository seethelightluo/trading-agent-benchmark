import pandas as pd, os
d='../persistent/stock_data'
df=pd.read_csv(os.path.join(d,'SPX.csv'), parse_dates=['date'])
print('SPX rows total:', len(df), 'min:', df.date.iloc[0], 'max:', df.date.iloc[-1])
cut = pd.Timestamp('2034-07-05')
sub = df[df.date<=cut]
print('rows <=2034-07-05:', len(sub), 'last:', sub.date.iloc[-1])
# check trading day granularity
import json
td = json.load(open('../persistent/date.json'))['trading_days']
td = [t for t in td if t <= '2034-07-05']
print('trading_days thru visible:', len(td), 'last:', td[-1])
# check date alignment with CSV
sub2 = df[df.date<=cut]['date'].dt.strftime('%Y-%m-%d').tolist()
print('csv dates in trading_days:', sum(1 for x in sub2 if x in set(td)))
print('csv dates not in td:', [x for x in sub2 if x not in set(td)][:10])