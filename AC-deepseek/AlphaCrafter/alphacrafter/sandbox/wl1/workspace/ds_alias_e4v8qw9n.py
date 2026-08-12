import json
d = json.load(open('scripts/miner2_screen_all_v2_results.json'))
passed = [x for x in d if x.get('passed')]
print("passed candidates in screen_all_v2:", len(passed))
for x in passed:
    print(x['name'], "ic=%.4f icir=%.4f" % (x['ic1']['ic'], x['ic1']['icir']))
print()
# also check names list
print("candidate names:", [x['name'] for x in d])