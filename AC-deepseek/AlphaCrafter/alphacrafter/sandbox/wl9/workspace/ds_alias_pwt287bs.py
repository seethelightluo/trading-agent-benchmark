import json,base64
d=json.load(open('factors/beta_VIX_60.json'))
data=d['validation']['signal_artifact']['data']
print('prefix repr:',repr(data[:100]))
print('format:',d['validation']['signal_artifact']['format'])
try:
    dec=base64.b64decode(data)
    print('b64 decoded len',len(dec),'head bytes',dec[:10])
    import zlib
    try:
        z=zlib.decompress(dec)
        print('zlib ok len',len(z), z[:80])
    except Exception as e:
        print('zlib fail',e, dec[:40])
except Exception as e:
    print('b64 fail',e)