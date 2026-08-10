import re
p = 'scripts/miner_1_20260730_screen_round11_novel.py'
src = open(p).read()

old_fn = """def f_month_season_prior(df, s):
    \"\"\"Prior-year same-calendar-month avg monthly return (fully causal: only years < current).\"\"\"
    c = df['close']
    y = df.index.year.values
    m = df.index.month.values
    g = df.groupby([df.index.year, df.index.month])['close'].last()
    g = g.reset_index().rename(columns={'level_0': 'year', 'level_1': 'month', 'close': 'last_close'})
    g['prev_close'] = g['last_close'].shift(1)
    g['mret'] = g['last_close'] / g['prev_close'] - 1.0
    g['key'] = g['year'] * 12 + g['month']
    mret_by_key = dict(zip(g['key'], g['mret']))
    vals = np.full(len(df), np.nan)
    keys = y * 12 + m
    for i in range(len(df)):
        yrs = np.arange(2020, y[i])
        ks = yrs * 12 + m[i]
        arr = [mret_by_key.get(k) for k in ks]
        arr = [a for a in arr if a is not None and np.isfinite(a)]
        if len(arr) >= 1:
            vals[i] = float(np.mean(arr))
    return pd.Series(vals, index=df.index)"""

new_fn = """def f_month_season_prior(df, s):
    \"\"\"Prior-year same-calendar-month avg monthly return (fully causal: only years < current).\"\"\"
    y = df.index.year.values
    m = df.index.month.values
    g = pd.DataFrame({'year': y, 'month': m, 'close': df['close'].values})
    g = g.groupby(['year', 'month'])['close'].last().reset_index()
    g = g.rename(columns={'close': 'last_close'})
    g['prev_close'] = g['last_close'].shift(1)
    g['mret'] = g['last_close'] / g['prev_close'] - 1.0
    g['key'] = g['year'] * 12 + g['month']
    mret_by_key = dict(zip(g['key'], g['mret']))
    vals = np.full(len(df), np.nan)
    for i in range(len(df)):
        yrs = np.arange(int(y[i]) - 6, int(y[i]))
        ks = yrs * 12 + m[i]
        arr = [mret_by_key.get(k) for k in ks]
        arr = [a for a in arr if a is not None and np.isfinite(a)]
        if len(arr) >= 1:
            vals[i] = float(np.mean(arr))
    return pd.Series(vals, index=df.index)"""

if old_fn not in src:
    raise SystemExit('old block not found')
src = src.replace(old_fn, new_fn)
open(p, 'w').write(src)
print('patched OK')
