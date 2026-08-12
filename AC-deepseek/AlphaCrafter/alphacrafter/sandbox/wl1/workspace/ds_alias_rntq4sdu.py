import json, os
print("date.json:", open('../persistent/date.json').read()[:200])
print()
print("index_data files:", os.listdir('../persistent/index_data'))
print("stock_data sample:", os.listdir('../persistent/stock_data')[:30])