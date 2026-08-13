import json
d = json.load(open('scripts/miner_3_20320624_revalidate_results.json'))
print(f"{'factor':28s} {'ic':>8s} {'icir':>8s} {'hit':>6s} {'last250ic':>9s} {'last250icir':>10s} {'ok':>4s}")
for k,v in d.items():
    lr = v.get('regime',{}).get('last250',{})
    print(f"{k:28s} {v['ic']:8.4f} {v['icir']:8.4f} {v['hit']:6.3f} {lr.get('ic',0):9.4f} {lr.get('icir',0):10.4f} {str(v.get('ok')):>4s}")
