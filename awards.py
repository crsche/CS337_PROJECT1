import re
import json
from collections import Counter
from fuzzywuzzy import fuzz

def remove_symbols(s):
    r = []
    for w in s.split():
        w = w.strip()
        if not w:
            continue
        if w.lower() == 'rt':
            continue
        if w[0] in {'@', '#'}:
            continue
        r.append(w)
    return ' '.join(r)

def _norm(s):
    s = s.lower()
    s = re.sub(r'\s+', ' ', s.strip())
    s = s.strip(" ,.:;!?\"'()[]{}_/\\")
    s = re.sub(r'\s*-\s*', ' - ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _canon_award(s):
    s = s.replace(' tv ', ' television ')
    s = s.replace(' t.v. ', ' television ')
    s = s.replace(' motion picture ', ' motion picture ')
    s = s.replace(' mini series', ' mini-series')
    s = s.replace(' miniseries', ' mini-series')
    s = s.replace(' limited series', ' mini-series')
    s = s.replace(' foreign film', ' foreign language film')
    s = s.replace(' original soundtrack', ' original score')
    s = s.replace(' feature animation', ' animated feature film')
    s = s.replace(' animated feature', ' animated feature film')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _split_camel_hashtag(s):
    if not s or not s.startswith('Best'):
        return None
    parts = re.findall(r'[A-Z][a-z]*|[A-Z]+(?![a-z])', s)
    if len(parts) < 2:
        return None
    return 'best ' + ' '.join(p.lower() for p in parts[1:])

def _slice_window(t, start, maxlen=80):
    end = min(len(t), start + maxlen)
    seg = t[start:end]
    stop = list(re.finditer(r':|–|—|-|;|,|\.|!|\?| goes to| goes | winner | win | for | at ', seg))
    if stop:
        seg = seg[:stop[-1].start()]
    return seg

def _patterns():
    p = []
    p.append(re.compile(r'\b(best [a-z0-9 ,&/\'\-]+)', re.I))
    p.append(re.compile(r'\bfor\s+(best [a-z0-9 ,&/\'\-]+)', re.I))
    p.append(re.compile(r'\baward for\s+(best [a-z0-9 ,&/\'\-]+)', re.I))
    p.append(re.compile(r'\b(best [a-z0-9 ,&/\'\-]+?)\s+(?:goes to|goes|is|was|win|winner|wins)\b', re.I))
    p.append(re.compile(r'\b(best[ A-Za-z0-9,&/\-]+?)\s*(?:–|—|-)\s*[A-Za-z]+', re.I))
    return p

def _harvest_award_spans(t):
    out = []
    for rx in _patterns():
        for m in rx.finditer(t):
            g = m.group(1)
            if g:
                out.append(g)
    for m in re.finditer(r'(best[ A-Za-z0-9/&\-]{3,})', t, re.I):
        seg = _slice_window(t, m.start(), 90)
        out.append(seg)
    return out

def _hashtags_to_awards(t):
    out = []
    for tag in re.findall(r'#([A-Za-z][A-Za-z0-9]+)', t):
        s = _split_camel_hashtag(tag)
        if s:
            out.append(s)
    return out

def _merge(items, k=200):
    cnt = Counter(items).most_common(k)
    res = []
    for s, c in cnt:
        put = True
        for j in res:
            a, n = j
            if fuzz.token_sort_ratio(a, s) > 88 and fuzz.partial_ratio(a, s) > 88:
                if len(s) < len(a):
                    j[0] = s
                j[1] += c
                put = False
                break
        if put:
            if len(res) < 120:
                res.append([s, c])
            else:
                res.sort(key=lambda x: x[1])
                res[0] = [s, 1]
    res.sort(key=lambda x: x[1], reverse=True)
    return res

def _post_rules(s):
    s = _norm(_canon_award(s))
    s = s.replace(' best best ', ' best ')
    s = s.replace(' best the ', ' best ')
    s = re.sub(r'\bperformance in a\b', ' performance by an ', s)
    s = re.sub(r'\bperformance in\b', ' performance by an ', s)
    s = s.replace(' actor actress ', ' actor ',)
    s = s.replace(' actress actor ', ' actress ',)
    s = s.replace(' best tv ', ' best television ')
    s = s.replace(' comedy musical', ' comedy or musical')
    s = s.replace(' musical comedy', ' comedy or musical')
    s = s.replace(' best original soundtrack ', ' best original score ')
    s = s.replace(' screenplay motion picture', ' screenplay - motion picture')
    s = s.replace(' director motion picture', ' director - motion picture')
    s = s.replace(' original song motion picture', ' original song - motion picture')
    s = s.replace(' original score motion picture', ' original score - motion picture')
    s = s.replace(' best animated feature ', ' best animated feature film ')
    s = s.replace(' best foreign film ', ' best foreign language film ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _valid(s):
    if len(s) < 12:
        return False
    if not s.startswith('best ') and not s.endswith(' demille award'):
        return False
    if re.search(r'[!@#%&]|http', s):
        return False
    return True

def run_awards(year):
    with open('gg'+str(year)+'.json', encoding='utf-8') as f:
        data = json.load(f)

    pool = []
    for it in data:
        raw = it.get('text', '')
        tw = remove_symbols(raw)
        low = tw.lower()
        if 'best' in low or '#best' in raw:
            spans = _harvest_award_spans(low)
            tags = _hashtags_to_awards(raw)
            for s in spans + tags:
                s = _post_rules(s)
                if _valid(s):
                    pool.append(s)

    merged = _merge(pool, k=300)

    extras = []
    for it in data:
        t = remove_symbols(it.get('text','').lower())
        m = re.search(r'\bthe\s+([a-z0-9 ,\'\-]{8,30})\s+award\b', t)
        if m:
            s = _post_rules('cecil b. demille award' if 'demille' in m.group(1) else m.group(1) + ' award')
            if _valid(s):
                extras.append(s)

    out = []
    seen = set()
    for w, _ in merged:
        x = _post_rules(w)
        if x not in seen:
            seen.add(x)
            out.append(x)

    for x in extras:
        y = _post_rules(x)
        if y not in seen:
            seen.add(y)
            out.append(y)

    final = []
    seen2 = set()
    for s in out:
        if s not in seen2:
            seen2.add(s)
            final.append(s)

    aug = []
    for s in list(final):
        if ' actor ' in s and ' actress ' not in s:
            aug.append(s.replace(' actor ', ' actress '))
        if ' actress ' in s and ' actor ' not in s:
            aug.append(s.replace(' actress ', ' actor '))
    for a in aug:
        if a not in seen2:
            seen2.add(a)
            final.append(a)

    return final

if __name__ == '__main__':
    res = run_awards(2013)
    for x in res[:60]:
        print(x)
