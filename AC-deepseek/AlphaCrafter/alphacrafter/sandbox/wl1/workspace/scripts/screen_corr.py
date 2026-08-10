import json, glob, os, gzip, base64, numpy as np, pandas as pd

files = sorted([f for f in glob.glob('factors/*.json') if not f.endswith('.bak')])
registry = {}
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        fid = d.get('factor_id')
        if not fid:
            continue
        base = not any(t in os.path.basename(f) for t in ['20260810T', '20260811T'])
        if fid not in registry or base:
            registry[fid] = d
    except Exception as e:
        print("ERR", f, e)

def decode_signal(d):
    sa = d.get('signal_artifact', {})
    b64 = sa.get('data_b64')
    if not b64:
        return None, None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(gzip.decompress(raw), dtype=np.float32).reshape(sa['n_dates'], sa['n_symbols'])
    dates = pd.date_range(sa['date_start'], sa['date_end'], periods=sa['n_dates']) if sa['n_dates'] > 1 else [pd.Timestamp(sa['date_start'])]
    return pd.DataFrame(arr, index=dates, columns=sa['symbols']), sa

sigs = {}
for fid, d in registry.items():
    s, sa = decode_signal(d)
    if s is None:
        print("NO SIGNAL for", fid)
        continue
    sigs[fid] = s

common = None
for fid, s in sigs.items():
    common = s.index if common is None else common.intersection(s.index)
print("Common dates:", len(common), common.min(), "->", common.max())

recent = common[-252:]
print("Recent window:", recent.min().date(), "->", recent.max().date())

def rank_stack(s):
    return s.rank(axis=1, pct=True)

stacked = {}
for fid, s in sigs.items():
    rs = rank_stack(s.loc[recent])
    stacked[fid] = rs.values.flatten()

fids = list(stacked.keys())
n = len(fids)
corr = pd.DataFrame(np.eye(n), index=fids, columns=fids)
for i in range(n):
    for j in range(i + 1, n):
        a = stacked[fids[i]]
        b = stacked[fids[j]]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() > 100:
            c = np.corrcoef(a[m], b[m])[0, 1]
        else:
            c = np.nan
        corr.iloc[i, j] = corr.iloc[j, i] = c

print("\n=== Pairwise rank-signal correlation (last 252d) ===")
print(corr.round(2).to_string())

# Quality metrics table
print("\n=== Factor quality (ic1, icir1, q=abs(ic)*abs(icir)) ===")
rows = []
for fid, d in registry.items():
    v = d.get('validation', {}).get('metrics', {})
    ic1 = v.get('ic1', np.nan)
    icir1 = v.get('icir1', np.nan)
    ic5 = v.get('ic5', np.nan)
    ic10 = v.get('ic10', np.nan)
    turn = v.get('turnover_10d', np.nan)
    q = abs(ic1) * abs(icir1)
    direction = d.get('expected_direction', 1)
    rows.append((fid, ic1, icir1, ic5, ic10, turn, q, direction))
for r in sorted(rows, key=lambda x: -x[6]):
    print("%-32s ic1=%7.4f icir1=%6.3f ic5=%7.4f ic10=%7.4f turn=%5.2f q=%7.5f dir=%d" % r)
