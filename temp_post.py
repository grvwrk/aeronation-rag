import json, urllib.request, urllib.error
url='http://127.0.0.1:8000/v1/chat'
data=json.dumps({'chat_id':'test','query':'Hello','category':'oil_gas'}).encode('utf-8')
req=urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print('status', r.status)
        print(r.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('HTTPError', e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    import traceback; traceback.print_exc()

