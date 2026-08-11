import json, glob, os
rows = []
for f in sorted(glob.glob('factors/*.json')):
    d = json.load(open(f))
    m = d.get('validation',{}).get('metrics',{})
    ba = d.get('benchmark_admission',{}).get('selected_metrics',{})
    fid = d.get('factor_id','?')
    cat = d.get('tags',['?'])
    ic = m.get('ic', ba.get('ic','?'))
    icir = m.get('icir', ba.get('icir','?'))
    q = ba.get('quality', None)
    rows.append((fid, str(cat[:2]), ic, icir, q, d.get('expected_direction')))
def fmt(x):
    try: return f"{float(x):.4f}"
    except: return str(x)
print(f"{'factor_id':26s} {'tags':34s} {'IC':>10s} {'ICIR':>10s} {'quality':>10s} dir")
for r in sorted(rows, key=lambda x: -(abs(x[2]) if isinstance(x[2],(int,float)) else 0)):
    print(f"{r[0]:26s} {str(r[1]):34s} {fmt(r[2]):>10s} {fmt(r[3]):>10s} {fmt(r[4]):>10s} {r[5]}")
