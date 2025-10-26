import json
import re
from collections import Counter

import spacy
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords





def _ensure_nltk():
    try:
        needed = [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("stopwords", "corpora/stopwords"),
        ]
        for pkg, path in needed:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass


_ensure_nltk()




def get_tweet_data(year):
    path = f"gg{year}.json"
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    texts = [t.get("text", "") for t in raw]

    cleaned = []
    for tw in texts:
        bits = []
        for w in tw.split():
            w = w.strip()
            if not w:
                continue
            if w.lower() == "rt":
                continue
            if w[0] in {"@", "#"}:
                continue
            bits.append(w)
        cleaned.append(" ".join(bits))

    return cleaned




def _safe_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {"the", "and", "of", "to", "a", "in", "for", "on", "with", "by", "from"}


def _simple_word_tokens(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())




def tweets_contain(category_name, tweets):
    sw = _safe_stopwords().union({"-", "performance", "comedy", "television"})
    toks = [t for t in _simple_word_tokens(category_name) if t not in sw]

    if not toks:
        return []

    patt = ".*?".join(map(re.escape, toks))
    rx = re.compile(patt, re.IGNORECASE)

    hits = [t for t in tweets if rx.search(t)]

    if "act" not in category_name.lower():
        hits = [x for x in hits if "act" not in x.lower()]

    return hits




def get_NNP(tweets_list):
    counter = Counter()
    for tw in tweets_list:
        toks = _simple_word_tokens(tw)
        try:
            pos = nltk.pos_tag(toks)
            for w, tag in pos:
                if tag in ("NNP", "NNPS"):
                    counter[w.lower()] += 1
        except LookupError:
            pass
    return dict(counter)




def count_name_mentions(tweets, candidates, fuzzy=False, min_ratio=90):
    counts = {c: 0 for c in candidates}
    for t in tweets:
        tl = t.lower()
        for c in candidates:
            cl = c.lower()
            if fuzzy:
                if fuzz.partial_ratio(cl, tl) >= min_ratio:
                    counts[c] += 1
            else:
                if cl in tl:
                    counts[c] += 1
    return counts




def get_person_names(tweets, nlp):
    names = set()

    m = Matcher(nlp.vocab)
    pat = [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}]
    try:
        m.add("FULL_NAME", [pat])
    except TypeError:
        m.add("FULL_NAME", None, pat)

    for t in tweets:
        doc = nlp(t)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.add(ent.text)

        for _, s, e in m(doc):
            names.add(doc[s:e].text)

    return list(names)




def remove_category_tokens(category, counts_dict):
    toks = _simple_word_tokens(category)
    toks += ["motion", "picture", "golden", "globes", "television", "series", "tv", "mini", "rt"]

    sw = _safe_stopwords()
    toks = [w for w in toks if w not in sw]

    for t in toks:
        for k in list(counts_dict.keys()):
            if t in k.lower():
                counts_dict[k] = 0

    return counts_dict




def get_top_percent(counts_dict, percentile):
    if not counts_dict:
        return []
    mx = max(counts_dict.values()) if counts_dict else 0
    if mx <= 0:
        return []
    th = mx * percentile
    return [k for k, v in counts_dict.items() if v > th]




def get_category_nominees(category, tweets, nlp):
    cat_tweets = tweets_contain(category, tweets)

    if ("actor" in category.lower()) or ("actress" in category.lower()):
        cands = get_person_names(cat_tweets, nlp)
        counts = count_name_mentions(cat_tweets, cands, fuzzy=False, min_ratio=90)
    else:
        counts = get_NNP(cat_tweets)

    counts = remove_category_tokens(category, counts)
    noms = get_top_percent(counts, percentile=0.85)
    return noms




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




def run_winners(year):
    tweets = get_tweet_data(year)

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError("spaCy model 'en_core_web_sm' not available.") from e

    if str(year) in {"2013"}:
        categories = AWARD_NAMES

    out = {}
    sep = " "

    for cat in categories:
        picks = get_category_nominees(cat, tweets, nlp)
        out[cat] = sep.join(picks)

    return out




if __name__ == "__main__":
    import sys
    y = sys.argv[1] if len(sys.argv) > 1 else "2013"
    res = run_winners(y)
    for k, v in res.items():
        print(f"{k}: {v}")
