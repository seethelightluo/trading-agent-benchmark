import csv
for sym in ['VIX','DXY','EURUSD']:
    p=f'../persistent/index_data/{sym}.csv'
    try:
        rows=list(csv.reader(open(p)))
        print(sym,'header:',rows[0])
        print(sym,'rows:',len(rows)-1)
        print('last 5:',rows[-5:])
    except Exception as e:
        print(sym,'ERR',e)
    print()