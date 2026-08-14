import json, zlib, base64, io, os, glob
import pandas as pd, numpy as np

CUT = pd.Timestamp('2035-04-19')
SD = '../persistent/stock_data'
watch = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

# ---- price panel ----
def load(p):
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df['close']

px = pd.DataFrame({s: load(os.path.join(SD, s+'.csv')) for s in watch}).sort_index()
px = px[px.index <= CUT]
print('price panel', px.shape, px.index.max().date())

# ---- factor panels ----
def decode(fpath):
    d = json.load(open(fpath))
    art = d.get('signal_artifact') or d.get('validation',{}).get('signal_artifact')
    if art is None:
        raise KeyError('no signal_artifact')
    raw = zlib.decompress(base64.b64decode(art['data']))
    df = pd.read_csv(io.BytesIO(raw))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df[df.index <= CUT]
    return df, d

factor_files = sorted(glob.glob('factors/*.json'))
panels = {}
meta = {}
for fp in factor_files:
    fid = os.path.basename(fp)[:-5]
    try:
        df, d = decode(fp)
        panels[fid] = df
        exp_dir = d.get('expected_direction', d.get('validation',{}).get('expected_direction', 1))
        meta[fid] = dict(file=fp, expected_direction=exp_dir,
                         ic_full=d.get('validation',{}).get('metrics',{}).get('ic'),
                         icir_full=d.get('validation',{}).get('metrics',{}).get('icir'),
                         turnover=d.get('validation',{}).get('metrics',{}).get('turnover_10d_rank'))
    except Exception as e:
        print('SKIP', fid, repr(e)[:120])

print('decoded factors:', len(panels))

# ---- forward returns on trading grid ----
all_dates = px.index
date_pos = {d: i for i, d in enumerate(all_dates)}

def fwd_ret(h):
    # fwd return h trading days ahead
    closes = px.values
    out = np.full(px.shape, np.nan)
    for i in range(len(all_dates) - h):
        out[i] = closes[i+h] / closes[i] - 1.0
    return pd.DataFrame(out, index=all_dates, columns=px.columns)

fwd = {h: fwd_ret(h) for h in [1,2,3,5,10]}

# ---- IC series ----
def ic_series(fid, h):
    panel = panels[fid]
    fr = fwd[h]
    common = panel.index.intersection(fr.index)
    if len(common) < 20:
        return None
    sig = panel.loc[common]
    rets = fr.loc[common]
    ics = []
    for t in common:
        s = sig.loc[t]
        r = rets.loc[t]
        m = s.notna() & r.notna()
        if m.sum() >= 8:
            ics.append((t, s[m].rank().corr(r[m].rank())))
    if not ics:
        return None
    out = pd.Series({t: v for t, v in ics}).sort_index()
    return out

print('\n%-34s %6s %6s %6s %6s %6s | %6s %6s %6s %6s %6s | %6s %6s' % (
    'factor', 'ic1','ic2','ic3','ic5','ic10', 'icir1','icir2','icir3','icir5','icir10', 'q10','hit10'))
rows = []
for fid in sorted(panels.keys()):
    row = {'fid': fid}
    for h in [1,2,3,5,10]:
        ics = ic_series(fid, h)
        if ics is None:
            row['ic%d'%h] = np.nan; row['icir%d'%h] = np.nan; row['hit%d'%h] = np.nan; row['n%d'%h] = 0
            continue
        # recent 60-date window
        recent = ics.iloc[-60:]
        ic = recent.mean()
        icir = recent.mean() / recent.std() if recent.std() > 0 else 0.0
        hit = (recent > 0).mean()
        row['ic%d'%h] = ic; row['icir%d'%h] = icir; row['hit%d'%h] = hit; row['n%d'%h] = len(recent)
    rows.append(row)
    print('%-34s %6.3f %6.3f %6.3f %6.3f %6.3f | %6.2f %6.2f %6.2f %6.2f %6.2f | %6.3f %6.2f' % (
        fid, row['ic1'], row['ic2'], row['ic3'], row['ic5'], row['ic10'],
        row['icir1'], row['icir2'], row['icir3'], row['icir5'], row['icir10'],
        abs(row['ic10'])*abs(row['icir10']), row['hit10']))

# save summary
with open('scripts/_screener_ic_summary.json','w') as f:
    json.dump([{k:(float(v) if isinstance(v,(int,float,np.floating)) and not isinstance(v,bool) else v) for k,v in r.items()} for r in rows], f, indent=1, default=str)
print('\nsaved scripts/_screener_ic_summary.json')
