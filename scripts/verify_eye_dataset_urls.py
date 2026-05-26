import urllib.request

urls = [
    'https://www5.cs.fau.de/research/data/fundus-images/',
    'https://drive5.com/DRIONS/',
    'https://www5.cs.fau.de/publications/retina-images/',
    'https://www3.cs.stonybrook.edu/~zeye/publications.html',
    'https://asiaa.co.jp/en/retinal-image-databases/',
]
headers = {'User-Agent': 'Mozilla/5.0'}

for u in urls:
    print('---', u)
    req = urllib.request.Request(u, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            content = r.read(10240).decode('utf-8', 'ignore')
            print('OK', len(content), 'bytes')
    except Exception as e:
        print('ERR', e)
