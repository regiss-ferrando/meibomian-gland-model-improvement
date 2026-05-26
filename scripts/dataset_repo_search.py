import urllib.request
import urllib.parse
import json

services = [
    ('Zenodo', 'https://zenodo.org/api/records/?q='),
    ('Figshare', 'https://api.figshare.com/v2/articles/search?search_term='),
]
terms = ['ocular', 'eye imaging', 'meibomian', 'ocular surface', 'fundus']
headers = {'User-Agent': 'Mozilla/5.0'}

for service, base in services:
    for term in terms:
        url = base + urllib.parse.quote(term) + '&size=10'
        print('---', service, term, '---')
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8', 'ignore')
            data = json.loads(text)
            if service == 'Zenodo':
                for hit in data.get('hits', {}).get('hits', [])[:5]:
                    print(hit.get('id'), hit.get('metadata', {}).get('title'), hit.get('links', {}).get('doi'))
            else:
                for hit in data[:5]:
                    print(hit.get('id'), hit.get('title'), hit.get('url'))
        except Exception as e:
            print('ERROR', e)
        print()