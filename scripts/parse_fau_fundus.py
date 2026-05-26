import urllib.request
import re
url = 'https://www5.cs.fau.de/research/data/fundus-images/'
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=30) as r:
    html = r.read().decode('utf-8', 'ignore')
print(html[:2000])
print('\n--- LINKS ---')
for m in re.findall(r'href=["\']([^"\']+)["\']', html):
    if 'http' in m or m.endswith('.zip') or 'DRIONS' in m or 'STARE' in m or 'RIGA' in m:
        print(m)