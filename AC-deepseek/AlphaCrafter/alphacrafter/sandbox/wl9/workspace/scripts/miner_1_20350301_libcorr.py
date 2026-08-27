import numpy as np, pandas as pd, json, base64, zlib, csv, io

DATA = '../persistent/stock_data'
WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = '2035-02-28'

close = {}
for s in WATCH:
    df = pd.read_csv(f'{DATA}/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF]
    df = df.set_index('date').sort_index()
    close[s] = df['close']
close = pd.DataFrame(close).replace(0, np.nan).ffill()
ret = close.pct_change()

def make_panel(fac):
    return fac

panels = {}
# rv_20 realized vol
panels['rv_20'] = ret.rolling(20).std()
# dd_from_high_60
panels['dd_from_high_60'] = close / close.rolling(60).max()

# ---- compare to existing library signals ----
def read_lib_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    try:
        sa = d['validation']['signal_artifact']
    except KeyError:
        return None
    data = sa['data']
    if data.startswith('base64:zlib:'):
        raw = zlib.decompress(base64.b64decode(data.split('base64:zlib:',1)[1])).decode()
    else:
        raw = base64.b64decode(data).decode()
    rd = csv.reader(io.StringIO(raw))
    header = next(rd)
    assets = header[1:]  # assume first col is date
    rows = []
    for row in rd:
        rows.append([row[0]] + [float(x) if x!='' else np.nan for x in row[1:]])
    df = pd.DataFrame(rows, columns=header)
    # pivot to date-indexed matrix
    if 'date' in df.columns:
        dfm = df.drop(columns=['date']).apply(pd.to_numeric, errors='coerce').astype(float)
        dfm.index = pd.to_datetime(df['date'])
    else:
        dfm = df.apply(pd.to_numeric, errors='coerce')
    # ensure columns = assets in WATCH order if present
    return dfm

LIB_IDS = ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']
libsigs = {}
for fid in LIB_IDS:
    try:
        p = read_lib_signal(fid)
        if p is not None:
            libsigs[fid] = p
    except Exception as e:
        print('lib read fail', fid, e)

for cand, pan in panels.items():
    print(f'=== candidate {cand} ===')
    for fid, ls in libsigs.items():
        # align columns by asset intersection
        common = [c for c in WATCH if c in ls.columns]
        if len(common) < 5: continue
        A = pan[common]; B = ls[common]
        idx = A.index.intersection(B.index)
        a = A.loc[idx].values.ravel(); b = B.loc[idx].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 500:
            r = np.corrcoef(a[m], b[m])[0,1]
            if abs(r) > 0.25:
                print(f'  vs {fid}: rho={r:.3f} n={m.sum()}')