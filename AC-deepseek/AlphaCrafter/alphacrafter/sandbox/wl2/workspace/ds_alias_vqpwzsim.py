with open('../persistent/stock_data/SPX.csv') as f:
    lines = f.readlines()
print("n lines:", len(lines))
print("first 3:", lines[:3])
print("last 3:", lines[-3:])
import json
print(json.load(open('../persistent/date.json')))
