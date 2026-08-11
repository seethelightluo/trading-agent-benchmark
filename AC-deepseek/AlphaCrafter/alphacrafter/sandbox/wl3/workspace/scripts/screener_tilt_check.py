import json, glob, os

files = sorted(glob.glob('factors/*.json'))
rows = []
for f in files:
    if f.endswith('factor_ensemble.json'):
        continue
    try:
        d = json.load(open(f))
    except Exception as e:
        print('ERR', f, e); continue
    fid = d.get('factor_id', os.path.basename(f)[:-5])
    m = d.get('validation', {}).get('metrics', {})
    ic = m.get('ic'); icir = m.get('icir')
    if ic is None or icir is None:
        rows.append((fid, None, None, None, None))
        continue
    q = abs(ic) * abs(icir)
    rows.append((fid, ic, icir, q, 1 if ic >= 0 else -1))

rows.sort(key=lambda r: -(r[3] if r[3] is not None else -1))
print(f"{'factor_id':<28}{'ic':>10}{'icir':>10}{'q=abs(ic)*abs(icir)':>22}{'dir':>5}")
for fid, ic, icir, q, direction in rows:
    if q is None:
        print(f"{fid:<28}{'MISSING':>10}")
    else:
        print(f"{fid:<28}{ic:>10.4f}{icir:>10.4f}{q:>22.6f}{direction:>5}")

ens = json.load(open('factors/factor_ensemble.json'))
print('\nCURRENT ENSEMBLE:')
tot = 0
for s in ens['selected_factors']:
    tot += s['weight']
    print(f"  {s['factor_id']:<28} w={s['weight']:.6f} dir={s['direction']}")
print(f"  sum={tot:.6f}")
