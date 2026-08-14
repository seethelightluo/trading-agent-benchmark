import json, glob, os

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if 'ensemble' not in f and not f.endswith('.bak')]

rows = []
for f in files:
    try:
        d = json.load(open(f))
        m = d['validation']['metrics']
        ic = m['ic']; icir = m['icir']
        q = abs(ic) * abs(icir)
        rows.append((d['factor_id'], ic, icir, m.get('ic_hit_ratio', 0), q,
                     d.get('expected_direction', ''), m.get('max_abs_library_correlation', 0),
                     m.get('turnover_10d_rank', 0), d['validation'].get('status','')))
    except Exception as e:
        rows.append((os.path.basename(f), None, None, None, None, str(e), None, None, 'ERR'))

rows.sort(key=lambda r: -(r[4] if r[4] is not None else -1))
print(f"{'factor_id':<28}{'ic':>8}{'icir':>8}{'hit':>7}{'q':>10}  {'dir':<8}{'maxcorr':>8}{'turn':>6}  status")
for r in rows:
    fid, ic, icir, hit, q, dirc, mc, tn, st = r
    if ic is None:
        print(f"{fid:<28}{'ERR':>8}  {dirc}")
        continue
    print(f"{fid:<28}{ic:>8.4f}{icir:>8.4f}{hit:>7.3f}{q:>10.5f}  {dirc:<8}{mc:>8.3f}{tn:>6.2f}  {st}")
