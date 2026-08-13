"""Trader 2031-06-30: block attribution for 06-16..06-30 cycle."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def get(a, n=40):
    df = None
    try:
        df = get_stock_daily_data(a, days=n)
    except Exception:
        pass
    if df is None or len(df) == 0:
        try:
            df = get_index_daily_data(a, days=n)
        except Exception:
            pass
    return df


acct = get_account_dict()
pos = {p['symbol']: p for p in acct.get('positions', [])}
net = acct.get('net_assets', 0)
print('net_assets:', round(net, 2))
print('cash:', round(acct.get('available_cash', 0), 2))

# current weights
mv_total = sum(p.get('market_value', 0) for p in acct.get('positions', []))
print('\ncurrent weights:')
for s in WATCH:
    p = pos.get(s)
    if p:
        w = p['market_value'] / mv_total if mv_total else 0
        print(f'  {s:10s} w={w:.4f} mv={p["market_value"]:.0f} pnl%={p.get("profit_loss_rate", 0):.4f}')
    else:
        print(f'  {s:10s} NO POSITION')

# block returns: last close visible (06-27) vs close at 06-16 decision (the close before block start)
print('\nblock returns (per asset, from 06-16 close to latest close):')
for s in WATCH:
    df = get(s, 40)
    if df is None or len(df) < 12:
        print(f'  {s:10s} NO DATA')
        continue
    df = df.sort_values('date')
    # find index of 2031-06-16 in df
    dates = [str(d)[:10] for d in df['date']]
    try:
        i = dates.index('2031-06-16')
    except ValueError:
        # use the last 10 rows as block proxy
        i = len(df) - 11
    c0 = float(df.iloc[i]['close'])
    c1 = float(df.iloc[-1]['close'])
    ret = (c1 / c0 - 1.0) * 100
    print(f'  {s:10s} ret={ret:+.2f}%  (c0={c0:.2f} -> c1={c1:.2f}, dates {dates[i]}..{dates[-1]})')

# implied block PnL attribution from current weights if weights unchanged (05-05 target)
print('\napprox weighted contrib (current weights * block ret):')
total = 0
for s in WATCH:
    df = get(s, 40)
    if df is None or len(df) < 12:
        continue
    df = df.sort_values('date')
    dates = [str(d)[:10] for d in df['date']]
    try:
        i = dates.index('2031-06-16')
    except ValueError:
        i = len(df) - 11
    c0 = float(df.iloc[i]['close'])
    c1 = float(df.iloc[-1]['close'])
    ret = c1 / c0 - 1.0
    p = pos.get(s)
    if p:
        w = p['market_value'] / mv_total if mv_total else 0
        contrib = w * ret * 100
        total += contrib
        print(f'  {s:10s} w={w:.4f} ret={ret*100:+.2f}% contrib={contrib:+.3f}%')
print('sum contrib (approx):', round(total, 3), '%')
