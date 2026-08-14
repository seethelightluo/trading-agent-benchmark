import csv, os, statistics

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
closes = {}
for a in assets:
    m = {}
    for r in csv.DictReader(open(f'../persistent/stock_data/{a}.csv')):
        if r['date'] <= '2035-04-27' and r['close'] not in ('', None):
            try:
                m[r['date']] = float(r['close'])
            except Exception:
                pass
    closes[a] = m

print("=== 1M (~21 obs) and 3M (~63 obs) returns per asset (through 2035-04-27) ===")
r1m, r3m = {}, {}
for a in assets:
    ds = sorted(closes[a].keys())
    ds = [d for d in ds if d <= '2035-04-27']
    if len(ds) < 63:
        print(a, 'insufficient obs', len(ds))
        continue
    c0_1m = closes[a][ds[-22]]
    c0_3m = closes[a][ds[-64]]
    c1 = closes[a][ds[-1]]
    r1m[a] = c1 / c0_1m - 1
    r3m[a] = c1 / c0_3m - 1
    print(f"{a:10s} 1M={r1m[a]*100:+7.2f}%  3M={r3m[a]*100:+7.2f}%")

active = [a for a in assets if a in r1m]
# frozen detection: last-21-obs return exactly 0 AND 3M return 0
frozen = [a for a in active if abs(r1m[a]) < 1e-9 and abs(r3m[a]) < 1e-9]
live = [a for a in active if a not in frozen]
print('\nfrozen (flat price) names:', frozen)
print('live names:', live)
rs = [r1m[a] for a in live]
print(f'1M cross-sectional mean={statistics.mean(rs)*100:+.2f}%  stdev={statistics.pstdev(rs)*100:.2f}%  '
      f'max={max(rs)*100:+.2f}%  min={min(rs)*100:+.2f}%  spread={(max(rs)-min(rs))*100:.2f}pp')
rs3 = [r3m[a] for a in live]
print(f'3M cross-sectional mean={statistics.mean(rs3)*100:+.2f}%  stdev={statistics.pstdev(rs3)*100:.2f}%  '
      f'spread={(max(rs3)-min(rs3))*100:.2f}pp')
order = sorted(live, key=lambda a: r3m[a], reverse=True)
print('3M ranking (live):')
for a in order:
    print(f"  {a:10s} 3M={r3m[a]*100:+7.2f}%  1M={r1m[a]*100:+7.2f}%")

# VIX regime from observation file
print('\n=== VIX (observation) recent ===')
vix = [r for r in csv.DictReader(open('../persistent/index_data/VIX.csv')) if r['date'] <= '2035-04-27']
vix = vix[-70:]
for r in vix[-12:]:
    print(r['date'], r['close'])
vals = [float(r['close']) for r in vix if r['close'] not in ('', None)]
print('VIX last-70d: min', min(vals), 'max', max(vals), 'last', vals[-1])
