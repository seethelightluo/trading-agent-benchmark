from pathlib import Path
import pandas as pd
STOCK_DIR = Path('../persistent/stock_data')
print(sorted(os.listdir(STOCK_DIR))[:50] if False else sorted([p.name for p in STOCK_DIR.iterdir()]))
print("---")
import os
print(sorted(os.listdir('../persistent/stock_data'))[:60])