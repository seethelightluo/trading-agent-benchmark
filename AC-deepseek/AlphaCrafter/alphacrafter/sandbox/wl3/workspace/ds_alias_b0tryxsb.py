import os, json
p = 'scripts/miner_1_20280810_revalidate_drift.json'
if os.path.exists(p):
    d = json.load(open(p))
    for k, v in d.items():
        print(k, "| warm ic=%.4f icir=%.4f" % (v.get('warm',{}).get('ic',0), v.get('warm',{}).get('icir',0)),
              "| oos ic=%.4f icir=%.4f" % (v.get('oos',{}).get('ic',0), v.get('oos',{}).get('icir',0)),
              "| recent ic=%.4f icir=%.4f" % (v.get('recent',{}).get('ic',0), v.get('recent',{}).get('icir',0)),
              "| verdict:", v.get('verdict'))
else:
    print("no json; listing script dir for outputs")
    print(sorted(os.listdir('scripts'))[-15:])