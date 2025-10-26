import json
import re
from collections import Counter, defaultdict

import spacy
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz, process
import nltk
from nltk.corpus import stopwords    
import sys


def _ensure_nltk():
    try:
        need = [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("stopwords", "corpora/stopwords"),
        ]
        for pkg, path in need:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass

_ensure_nltk()

STOP = set(stopwords.words("english")) if hasattr(stopwords, "words") else set()
STOP.update({"-", "–", "—", "performance", "television", "tv", "series", "motion", "picture", "award"})

AWARD_NAMES = [
    'cecil b. demille award', 'best motion picture - drama',
    'best performance by an actress in a motion picture - drama',
    'best performance by an actor in a motion picture - drama',
    'best motion picture - comedy or musical',
    'best performance by an actress in a motion picture - comedy or musical',
    'best performance by an actor in a motion picture - comedy or musical',
    'best animated feature film', 'best foreign language film',
    'best performance by an actress in a supporting role in a motion picture',
    'best performance by an actor in a supporting role in a motion picture',
    'best director - motion picture', 'best screenplay - motion picture',
    'best original score - motion picture', 'best original song - motion picture',
    'best television series - drama',
    'best performance by an actress in a television series - drama',
    'best performance by an actor in a television series - drama',
    'best television series - comedy or musical',
    'best performance by an actress in a television series - comedy or musical',
    'best performance by an actor in a television series - comedy or musical',
    'best mini-series or motion picture made for television',
    'best performance by an actress in a mini-series or motion picture made for television',
    'best performance by an actor in a mini-series or motion picture made for television',
    'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
    'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television'
]


def get_top_percent(d, percentile=None, pct=None):
    if not d:
        return []
    frac = 0.85
    if percentile is not None:
        frac = float(percentile)
    elif pct is not None:
        frac = float(pct)
    m = max(d.values())
    if m <= 0:
        return []
    thr = m * frac
    return [k for k, v in d.items() if v > thr]


def _simple_tokens(s):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s.lower())


def _clean_text(t):
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'&amp;', '&', t)
    parts = []
    for w in t.split():
        if w.lower() == "rt": 
            continue
        if w[:1] in {"@", "#"}: 
            continue
        parts.append(w)
    return " ".join(parts).strip()


def get_tweet_data(year):
    path = f"gg{year}.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [_clean_text(t.get("text", "")) for t in raw]


def tweets_contain(category, tweets):
    toks = [w for w in _simple_tokens(category) if w not in STOP]
    if not toks:
        return []
    rg = re.compile(".*?".join(map(re.escape, toks)), re.IGNORECASE)
    hits = [t for t in tweets if rg.search(t)]
    if "act" not in category.lower():
        hits = [t for t in hits if "act" not in t.lower()]  # reduce bleed
    return hits


def remove_category_tokens(category, counts_dict):
    bad = set(_simple_tokens(category))
    bad.update({"motion", "picture", "golden", "globe", "globes", "television", "series", "tv", "mini"})
    for k in list(counts_dict.keys()):
        if any(b in k.lower() for b in bad):
            counts_dict[k] = 0
    return counts_dict


def _build_matcher(nlp):
    m = Matcher(nlp.vocab)
    pat = [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}]
    try:
        m.add("FULL_NAME", [pat])
    except TypeError:
        m.add("FULL_NAME", None, pat)

    return m


def _person_candidates(tweets, nlp, matcher):
    names = set()
    for t in tweets:
        doc = nlp(t)
        for e in doc.ents:
            if e.label_ == "PERSON":
                names.add(e.text)
        for _, s, e in matcher(doc):
            span = doc[s:e].text
            if len(span.split()) >= 2:
                names.add(span)
    return list(names)


def _title_candidates(tweets, nlp):
    titles = set()
    for t in tweets:
        doc = nlp(t)
        for e in doc.ents:
            if e.label_ in {"WORK_OF_ART", "ORG"}:
                titles.add(e.text)

    cnt = Counter()
    for t in tweets:
        tokens = nltk.pos_tag(_simple_tokens(t))
        for w, tag in tokens:
            if tag in ("NNP", "NNPS"):
                cnt[w.lower()] += 1
    for k, v in cnt.items():
        if v >= 3:
            titles.add(k)
    return list(titles)


def count_name_mentions(tweets, candidates, fuzzy=True, min_ratio=85):
    counts = {c: 0 for c in candidates}
    for t in tweets:
        tl = t.lower()
        if fuzzy:
            # token_set helps with variations
            for c in candidates:
                if fuzz.token_set_ratio(c.lower(), tl) >= min_ratio:
                    counts[c] += 1
        else:
            for c in candidates:
                if c.lower() in tl:
                    counts[c] += 1
    return counts


def get_category_nominees(category, tweets, nlp, matcher):
    cat_tweets = tweets_contain(category, tweets)
    if not cat_tweets:
        return []

    if ("actor" in category.lower()) or ("actress" in category.lower()):
        cands = _person_candidates(cat_tweets, nlp, matcher)
        counts = count_name_mentions(cat_tweets, cands, fuzzy=True, min_ratio=85)
    else:
        cands = _title_candidates(cat_tweets, nlp)
        counts = count_name_mentions(cat_tweets, cands, fuzzy=True, min_ratio=80)

    counts = remove_category_tokens(category, counts)
    top = get_top_percent(counts, percentile=0.5)

    if not top:
        top = [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4]]

    return [t.strip().lower() for t in top][:4]


def run_nominees(year):
    tweets = get_tweet_data(year)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    matcher = _build_matcher(nlp)


    categories = AWARD_NAMES if str(year) in {"2013"} else AWARD_NAMES
    out = {}
    for cat in categories:
        noms = get_category_nominees(cat, tweets, nlp, matcher)
        while len(noms) < 4:
            noms.append("l")
        out[cat] = noms
    return out


if __name__ == "__main__":
    yr = sys.argv[1] if len(sys.argv) > 1 else "2013"
    res = run_nominees(yr)
    for k, v in res.items():
        print(k, ":", ", ".join(v))
