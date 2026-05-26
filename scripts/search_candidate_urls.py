import urllib.request
import urllib.parse
import re

queries = [
    'public ocular dataset figshare',
    'meibomian gland dataset public download',
    'ocular surface imaging dataset open access',
    'fundus image dataset new dataset 2024',
]
headers = {'User-Agent': 'Mozilla/5.0'}

for q in queries:
    print('--- QUERY:', q, '---')
    url = 'https://www.bing.com/search?q=' + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode('utf-8', 'ignore')
    urls = re.findall(r'<a[^>]+href="(https?://[^"#]+)"', html)
    seen = set()
    count = 0
    for u in urls:
        if u not in seen and 'bing.com' not in u and 'microsoft.com' not in u:
            seen.add(u)
            print(u)
            count += 1
            if count >= 30:
                break
    print()
