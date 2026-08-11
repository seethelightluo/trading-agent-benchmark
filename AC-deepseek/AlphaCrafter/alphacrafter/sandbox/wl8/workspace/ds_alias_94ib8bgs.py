import json, pandas as pd
# trading calendar check between 2026-07-16 and 2026-07-30
from alphacrafter.sim.utils import get_stock_daily_data
df = get_stock_daily_data('SPX', days=40)
df = df.sort_values('date')
cal = [d.strftime('%Y-%m-%d') for d in df['date']]
print('recent calendar:', cal[-20:])
idx16 = cal.index('2026-07-16') if '2026-07-16' in cal else None
idx30 = cal.index('2026-07-30') if '2026-07-30' in cal else None
print('idx 07-16:', idx16, 'idx 07-30:', idx30, 'days between (exclusive):', (idx30-idx16) if idx16 is not None and idx30 is not None else None)
print('07-29 in cal:', '2026-07-29' in cal)
