import json, base64, zlib, io
import pandas as pd, numpy as np

def load_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    sa = d.get('signal_artifact', {})
    fmt = sa.get('format', '')
    if fmt == 'base64:zlib:csv':
        raw = base64.b64decode(sa['data'])
        raw = zlib.decompress(raw)
        df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        return df
    elif fmt == 'panel_json_v1':
        dates = sa['dates']; cols = sa['columns']
        mat = sa.get('matrix')
        if mat:
            return pd.DataFrame(np.array(mat), index=pd.to_datetime(dates), columns=cols).astype(float)
    return None

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px = {}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[a] = df['close'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[px.index <= '2031-12-12']

for fid in ['vol_adj_mom_accel_20x60', 'dn_mkt_beta_60d', 'rate_beta_cn10y_60d']:
    sig = load_signal(fid)
    if sig is None:
        print(fid, 'no signal decoded')
        continue
    sig = sig.sort_index()
    common = sig.index.intersection(px.index)
    s = sig.loc[common]
    p = px.loc[common]
    fr = (p.shift(-11) / p.shift(-1) - 1)  # forward 10d return, next-day start
    ics = []
    for dt in s.index:
        if dt not in fr.index:
            continue
        x = s.loc[dt].astype(float)
        y = fr.loc[dt]
        m = pd.concat([x, y], axis=1).dropna()
        if len(m) >= 8:
            ic = m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank())
            if pd.notna(ic):
                ics.append((dt, ic))
    ics = pd.DataFrame(ics, columns=['date', 'ic']).set_index('date')
    print(f'=== {fid} ===  total_ic_dates={len(ics)}')
    for win in [60, 120, 250]:
        sub = ics.tail(win)
        if len(sub) > 10:
            print(f"  recent{win}d: IC={sub['ic'].mean():+.4f} ICIR={sub['ic'].mean()/sub['ic'].std():+.3f} n={len(sub)} hit={(sub['ic']>0).mean():.2f}")
    # last 20 IC dates
    print('  last-20 ICs:', [round(v, 3) for v in ics['ic'].tail(20).tolist()])
    print()
