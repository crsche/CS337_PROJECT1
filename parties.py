import re
import sys
import json
from collections import defaultdict, Counter

import spacy
from nltk import download as _nltk_dl
from nltk.sentiment import SentimentIntensityAnalyzer

RE_SPACES = re.compile(r"\s+")
RE_CAMEL = re.compile(r"[A-Z][a-z]*|[A-Z]+(?![a-z])|[a-z]+")
RE_HASHTAGS = re.compile(r"#([A-Za-z][A-Za-z0-9\-]+)")
RE_AFTER_PARTY = re.compile(r"\bafter\s*party\b")
RE_STRIP = re.compile(r"[ ,.:;!?\"'()\[\]{}\_/\\]+$")


def _ensure_vader():
    try:
        SentimentIntensityAnalyzer()
        return
    except Exception:
        pass
    try:
        _nltk_dl("vader_lexicon", quiet=True)
    except Exception:
        pass

def _strip_marks(s):
    out = []
    for w in s.split():
        w = w.strip()
        if not w or w.lower() == "rt":
            continue
        if w[0] in {"@", "#"}:
            if w.lower().startswith("#best"):
                out.append(w)
            continue
        out.append(w)
    return " ".join(out)


def _norm(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s.strip())
    s = s.strip(" ,.:;!?\"'()[]{}_/\\")
    return s

def _split_camel(tag):
    parts = RE_CAMEL.findall(tag)
    return " ".join(p.lower() for p in parts) if parts else None



def _hashtag_parties(raw_text):
    out = []
    for tag in RE_HASHTAGS.findall(raw_text):
        low = tag.lower()
        if "party" not in low:
            continue
        s = _split_camel(tag) or low
        s = s.replace("-", " ")
        if "party" in s:
            out.append(_norm(s))
    return out


def _around_party_phrase(doc):
    names = set()
    tokens = [(t.text.lower(), t.lemma_.lower(), t.pos_) for t in doc]
    n = len(tokens)
    for i, (txt, lem, pos) in enumerate(tokens):
        if txt in {"party", "after-party", "afterparty"} or lem == "party":
            # Expand backward and forward
            start = i
            while start - 1 >= 0 and tokens[start - 1][2] in {"PROPN", "NOUN", "ADJ"}:
                start -= 1
            end = i + 1
            while end < n and tokens[end][2] in {"PROPN", "NOUN"}:
                end += 1
            span = _norm(doc[start:end].text)
            if "party" in span:
                names.add(span)

            # Find "at" phrases
            for j in range(max(0, i - 5), min(n, i + 6)):
                if tokens[j][0] == "at":
                    s = j + 1
                    e = s
                    while e < n and tokens[e][2] in {"PROPN", "NOUN"}:
                        e += 1
                    span2 = _norm(doc[s:e].text + " party")
                    if "party" in span2 and len(doc[s:e]):
                        names.add(span2)
                    break
    return list(names)



def _ner_party_brands(doc):
    names = set()
    orgs = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "WORK_OF_ART", "EVENT"}]
   
   
    for o in orgs:
        s = _norm(o + " party")
        if "party" in s:
            names.add(s)

    return list(names)

def _extract_parties(nlp, raw_text):
    t = _strip_marks(raw_text)
    low = t.lower()
    if not any(k in low for k in ("party", "afterparty", "after-party", "#party")):
        return []
    
    out = []
    out += _hashtag_parties(raw_text)
    doc = nlp(t)
    out += _around_party_phrase(doc)
    out += _ner_party_brands(doc)
    out = [re.sub(r"\bafter\s*party\b", "after party", x) for x in out]
    out = [x.replace(" after party", " after party") for x in out]
    out = [re.sub(r"\s+", " ", x).strip() for x in out]
    out = [x for x in out if len(x) >= 6 and "party" in x]

    return list(dict.fromkeys(out))

def _merge_labels(labels):
    base = [re.sub(r"\s+", " ", s.replace(" after party", " after party")).strip() for s in labels]
    counts = Counter(base)
    merged = []
    for s, c in counts.most_common():
        for m in merged:
            a, n = m
            if (
                a == s or
                (a.endswith(" after party") and s.startswith(a[:-12])) or
                (s.endswith(" after party") and a.startswith(s[:-12]))
            ):
                m[1] += c
                break
        else:
            merged.append([s, c])
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

def run_parties(year, top_k=10):
    with open(f"gg{year}.json", encoding="utf-8") as f:
        data = json.load(f)

    nlp = spacy.load("en_core_web_sm")
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()

    party_map = defaultdict(list)
    for it in data:
        raw = it.get("text", "")
        labs = _extract_parties(nlp, raw)
        if not labs:
            continue
        for lb in labs:
            party_map[lb].append(raw)

    merged = _merge_labels(list(party_map.keys()))
    rank = [p for p, _ in merged[: max(top_k, 1)]]

    stats = []
    for p in rank:
        tweets = []
        for key in party_map:
            if key == p or key.startswith(p.replace(" after party", "")) or p.startswith(key.replace(" after party", "")):
                tweets.extend(party_map[key])
        seen_t = set()
        dedup = []


        for tw in tweets:
            tnorm = _norm(tw)
            if tnorm in seen_t:
                continue
            seen_t.add(tnorm)
            dedup.append(tw)
        pos = neg = neu = 0
        comp = 0.0
        for tw in dedup:
            s = sia.polarity_scores(tw)
            comp += s["compound"]
            if s["compound"] >= 0.2:
                pos += 1
            elif s["compound"] <= -0.2:
                neg += 1
            else:
                neu += 1
        total = max(1, len(dedup))
        stats.append({
            "party": p,
            "mentions": len(dedup),
            "pos": pos,
            "neg": neg,
            "neu": neu,
            "pos_pct": round(100.0 * pos / total, 1),
            "neg_pct": round(100.0 * neg / total, 1),
            "neu_pct": round(100.0 * neu / total, 1),
            "avg_compound": round(comp / total, 3),
            "samples": dedup[:3]
        })


    stats.sort(key=lambda x: (x["mentions"], x["avg_compound"]), reverse=True)

    print(f"Top {min(top_k, len(stats))} Golden Globes party conversations in {year}")
    print("")
    for i, row in enumerate(stats[:top_k], 1):
        print(f"{i}. {row['party']} — mentions: {row['mentions']} | +{row['pos_pct']}% / -{row['neg_pct']}% / ~{row['neu_pct']}% | avg={row['avg_compound']}")
        for s in row["samples"]:
            print(f"   · {s}")
        print("")

    return stats




if __name__ == "__main__":
    y = sys.argv[1] if len(sys.argv) > 1 else "2013"
    run_parties(y)
